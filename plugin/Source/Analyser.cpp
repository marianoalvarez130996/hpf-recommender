#include "Analyser.h"

#include <hpf/hpf.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace hpfplugin {

namespace {

juce::String number(double value, int decimals) { return juce::String(value, decimals); }

// los textos con acentos tienen que entrar como UTF-8
juce::String utf8(const char* text) { return juce::String::fromUTF8(text); }

// 1 y 3 se escriben sin decimales; 0.5, con uno
juce::String compact(double value)
{
    return std::abs(value - std::round(value)) < 1e-9 ? juce::String(juce::roundToInt(value)) : number(value, 1);
}

juce::String describe(const hpf::Recommendation& r, double seconds, double sampleRate, int channels,
                      double droppedSeconds)
{
    juce::StringArray lines;
    if (!r.cutoff_hz || !r.keep_hz || !r.loss_budget_db) {
        lines.add(utf8("No filter recommendation."));
    } else {
        // una línea por parámetro del filtro, con la razón de cada decisión
        // el filtro fijo de 20 Hz trabaja sin anunciarse: la ventana habla solo del filtro recomendado
        lines.add(r.active ? utf8("Filter: on (there is low-frequency noise to remove)")
                           : utf8("Filter: off (there is nothing measurable to remove)"));
        const juce::String note = r.f0_note ? " (" + juce::String(*r.f0_note) + ")" : juce::String();
        juce::String what = "the lower edge of the spectrum";
        if (r.source == "f0")
            what = utf8("the lowest note") + note;
        else if (r.source == "mixto")
            what = utf8("the lowest note") + note + utf8(", found in few frames and below the spectral edge,");
        lines.add("Cutoff: " + number(*r.cutoff_hz, 1) + utf8(" Hz, the highest that keeps ") + what + " at "
                  + number(*r.keep_hz, 1) + " Hz within " + compact(*r.loss_budget_db)
                  + utf8(" dB of loss"));
        juce::String why = utf8("the gentlest that brings the low-frequency noise down to the target");
        if (!r.rejection_met)
            why = (r.source == "espectro" && r.order == hpf::edge_max_order
                       ? utf8("the steepest allowed on a spectral edge; even so the low-frequency noise stays ")
                       : utf8("the steepest available; even so the low-frequency noise stays "))
                  + number(r.rejection_short_db, 0) + " dB above the target";
        else if (!r.active)
            why = utf8("the gentlest, in case you want to filter anyway");
        lines.add("Slope: " + juce::String(r.slope_db_oct) + " dB/oct, " + why);
        lines.add(utf8("Curve: Butterworth, the flattest: no boost or ripple above the cutoff"));
    }
    if (r.loss_at_keep_db && r.phase_at_keep_deg && r.group_delay_ms)
        lines.add(utf8("At that frequency: ") + number(*r.loss_at_keep_db, 2) + " dB of loss, " + number(*r.phase_at_keep_deg, 1)
                  + " degrees of phase, " + number(*r.group_delay_ms, 1) + " ms of group delay");
    if (r.hum) {
        juce::StringArray harmonics;
        for (double f : r.hum->harmonics_hz) harmonics.add(number(f, 0));
        lines.add("Mains hum at " + number(r.hum->base_hz, 0) + " Hz (" + harmonics.joinIntoString(", ") + " Hz)");
    }
    for (const auto& warning : r.warnings) {
        juce::String w = juce::String::fromUTF8(warning.c_str());
        if (r.base_hz && w.startsWith("DC offset")) continue;   // el filtro fijo ya lo quita
        if (r.base_hz) {                                        // sin mencionar el filtro fijo
            const int at = w.indexOf(" (what is left after the fixed ");
            if (at >= 0) w = w.substring(0, at) + w.substring(w.indexOf(at, " Hz filter)") + 11);
        }
        lines.add("Note: " + w);
    }
    lines.add("Analysed: " + number(seconds, 1) + " s at " + number(sampleRate / 1000.0, 1) + " kHz, "
              + juce::String(channels) + (channels == 1 ? " channel" : " channels"));
    if (droppedSeconds > 0.0)
        lines.add("Warning: " + number(droppedSeconds, 2) + " s of audio were lost while listening.");
    return lines.joinIntoString("\n");
}

}  // namespace

// Un análisis a la vez en todo el proceso: con el plugin en muchas pistas, todas terminan de escuchar
// juntas y cada análisis copia la pista a doble precisión.
static std::mutex& analysisLock()
{
    static std::mutex m;
    return m;
}

Analyser::Analyser(std::function<void()> finished) : juce::Thread("HPF! analysis"), onFinished(std::move(finished))
{
    startThread(juce::Thread::Priority::low);
}

Analyser::~Analyser()
{
    stopThread(-1);   // hpf::recommend no se puede interrumpir: se espera a que termine
}

void Analyser::prepare(double sampleRate, int numChannels)
{
    {
        const std::lock_guard<std::mutex> lock(configMutex);
        drainLocked();
        numChannels = juce::jlimit(1, 2, numChannels);
        const bool changed = !juce::exactlyEqual(sampleRate, rate.load()) || numChannels != channels;
        bool resume = false;
        if (changed) {
            // otro formato a mitad de la escucha: un render que empieza o termina a otra frecuencia,
            // o el proyecto que cambió. Lo escuchado sigue siendo válido a su propia frecuencia.
            const bool wasListening = listening.exchange(false);
            if (wasListening && static_cast<double>(capturedSamples.load()) >= minSeconds * captureRate.load()) {
                analyseRequested = true;   // alcanza: se analiza ya, antes de descartar nada
            } else if (!analyseRequested.load()) {
                // no alcanza (o no se escuchaba): se empieza de cero en el formato nuevo
                captured.assign(static_cast<std::size_t>(numChannels), {});
                identicalChannels = true;
                capturedSamples = 0;
                captureRate = sampleRate;
                resume = wasListening;
            }
        }
        channels = numChannels;
        rate = sampleRate;
        const int capacity = juce::nextPowerOfTwo(juce::jmax(1 << 14, static_cast<int>(sampleRate)));   // ~1 s
        queue.assign(static_cast<std::size_t>(channels), std::vector<float>(static_cast<std::size_t>(capacity)));
        fifo.setTotalSize(capacity);
        fifo.reset();
        if (resume) listening.store(true, std::memory_order_release);
    }
    notify();
}

void Analyser::push(const juce::AudioBuffer<float>& buffer, bool mayWait) noexcept
{
    if (!listening.load(std::memory_order_acquire) || channels == 0) return;
    const int n = buffer.getNumSamples();
    // fuera de tiempo real el host entrega bloques más rápido que el reloj: se espera (hasta 5 s) a
    // que el hilo de análisis vacíe la cola
    for (int waited = 0; mayWait && fifo.getFreeSpace() < n && waited < 5000; ++waited) {
        notify();
        juce::Thread::sleep(1);
    }
    int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
    fifo.prepareToWrite(n, start1, size1, start2, size2);
    for (int ch = 0; ch < channels; ++ch) {
        // si el host entrega menos canales, se repite el primero para que todos midan lo mismo
        const float* src = buffer.getReadPointer(juce::jmin(ch, buffer.getNumChannels() - 1));
        float* dst = queue[static_cast<std::size_t>(ch)].data();
        std::copy(src, src + size1, dst + start1);
        std::copy(src + size1, src + size1 + size2, dst + start2);
    }
    fifo.finishedWrite(size1 + size2);
    if (size1 + size2 < n) droppedSamples.fetch_add(n - size1 - size2, std::memory_order_relaxed);
}

void Analyser::startListening()
{
    if (listening.load() || stage.load() == Stage::analysing) return;
    {
        const std::lock_guard<std::mutex> lock(configMutex);
        fifo.reset();
        // con el formato actual del host: una captura anterior pudo quedar en otro
        captured.assign(static_cast<std::size_t>(juce::jmax(1, channels)), {});
        identicalChannels = true;
        capturedSamples = 0;
        droppedSamples = 0;
        captureRate = rate.load();
    }
    stage = Stage::listening;
    listening.store(true, std::memory_order_release);
    notify();
}

void Analyser::stopAndAnalyse()
{
    {
        // con el mismo cerrojo que prepare(), que mira estas dos banderas para no descartar una
        // captura que ya se pidió analizar
        const std::lock_guard<std::mutex> lock(configMutex);
        if (!listening.exchange(false)) return;
        analyseRequested = true;
    }
    notify();
}

double Analyser::getCapturedSeconds() const noexcept
{
    const double sr = captureRate.load();
    return sr > 0.0 ? static_cast<double>(capturedSamples.load()) / sr : 0.0;
}

std::optional<Analysis> Analyser::getResult() const
{
    const std::lock_guard<std::mutex> lock(resultMutex);
    return result;
}

void Analyser::restoreResult(Analysis analysis)
{
    // una sesión guardada trae su análisis: lo que se estaba escuchando ya no hace falta
    listening = false;
    {
        const std::lock_guard<std::mutex> lock(resultMutex);
        result = std::move(analysis);
    }
    ++resultVersion;
    if (stage.load() == Stage::idle || stage.load() == Stage::listening) stage = Stage::done;
}

void Analyser::run()
{
    long long lastCount = -1;
    juce::uint32 lastChange = 0;
    while (!threadShouldExit()) {
        {
            const std::lock_guard<std::mutex> lock(configMutex);
            drainLocked();
        }
        // si deja de llegar audio (se detuvo el transporte o terminó el render) con al menos
        // minSeconds escuchados, se analiza sin esperar al botón
        if (listening.load()) {
            const auto now = juce::Time::getMillisecondCounter();
            const long long count = capturedSamples.load();
            if (count != lastCount) {
                lastCount = count;
                lastChange = now;
            } else if (static_cast<double>(count) >= minSeconds * captureRate.load() && now - lastChange >= idleMs) {
                stopAndAnalyse();
            }
        } else {
            lastCount = -1;
        }
        if (analyseRequested.exchange(false)) analyse();
        wait(listening.load() ? 20 : -1);   // sin escucha duerme hasta que lo despierten
    }
}

void Analyser::drainLocked()
{
    const int ready = fifo.getNumReady();
    if (ready <= 0 || captured.empty()) return;
    int start1 = 0, size1 = 0, start2 = 0, size2 = 0;
    fifo.prepareToRead(ready, start1, size1, start2, size2);
    // pista mono en un bus estéreo: mientras L y R sean idénticos se guarda solo L (la mitad de
    // memoria); en la primera diferencia R recibe la historia de L y desde ahí se guardan los dos
    if (captured.size() == 2 && identicalChannels) {
        const float* l = queue[0].data();
        const float* r = queue[1].data();
        const bool same = std::equal(l + start1, l + start1 + size1, r + start1)
                          && std::equal(l + start2, l + start2 + size2, r + start2);
        if (!same) {
            identicalChannels = false;
            captured[1] = captured[0];
        }
    }
    const std::size_t stored = (captured.size() == 2 && identicalChannels) ? 1 : captured.size();
    for (std::size_t ch = 0; ch < stored; ++ch) {
        const float* src = queue[ch].data();
        captured[ch].insert(captured[ch].end(), src + start1, src + start1 + size1);
        captured[ch].insert(captured[ch].end(), src + start2, src + start2 + size2);
    }
    fifo.finishedRead(size1 + size2);
    capturedSamples = static_cast<long long>(captured.front().size());
    if (listening.load() && capturedSamples.load() >= static_cast<long long>(maxSeconds * captureRate.load())) {
        listening = false;            // tope de memoria: se analiza lo escuchado
        analyseRequested = true;
    }
}

void Analyser::analyse()
{
    stage = Stage::analysing;
    hpf::Channels x;
    double sr = 0.0;
    {
        const std::lock_guard<std::mutex> lock(configMutex);
        drainLocked();
        sr = captureRate.load();   // la de lo escuchado, aunque el host ya haya cambiado
        // canales idénticos: se analiza uno solo, que es lo que de verdad contiene la pista
        const std::size_t keep = identicalChannels ? std::min<std::size_t>(captured.size(), 1) : captured.size();
        for (std::size_t ch = 0; ch < captured.size(); ++ch) {
            if (ch < keep) x.emplace_back(captured[ch].begin(), captured[ch].end());
            std::vector<float>().swap(captured[ch]);   // libera la memoria de la captura
        }
        identicalChannels = true;
        capturedSamples = 0;
    }
    Analysis analysis;
    Stage next = Stage::done;
    const double seconds = (x.empty() || sr <= 0.0) ? 0.0 : static_cast<double>(x.front().size()) / sr;
    try {
        if (seconds < 1.0) throw std::runtime_error("less than one second of audio was heard.");
        const std::lock_guard<std::mutex> one(analysisLock());
        const hpf::Recommendation r = hpf::recommend(x, sr);
        analysis.cutoffHz = r.cutoff_hz;
        analysis.order = r.order;
        analysis.active = r.active;
        analysis.summary = describe(r, seconds, sr, static_cast<int>(x.size()),
                                    static_cast<double>(droppedSamples.load()) / sr);
        analysis.json = juce::String::fromUTF8(hpf::to_json(r).c_str());
    } catch (const std::exception& e) {
        analysis.summary = "Could not analyse: " + juce::String::fromUTF8(e.what());
        next = Stage::failed;
    }
    {
        const std::lock_guard<std::mutex> lock(resultMutex);
        result = std::move(analysis);
    }
    ++resultVersion;
    stage = next;
    if (next == Stage::done && onFinished) onFinished();
}

}  // namespace hpfplugin
