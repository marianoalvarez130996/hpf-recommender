// Prueba el plugin compilado como lo usa un DAW, sin DAW: lo carga como VST3, simula un transporte que
// reproduce y se detiene, y un render fuera de tiempo real, y comprueba que escucha solo desde que se
// inserta, que analiza al detenerse, que aplica la recomendación (el mismo corte, pendiente y encendido
// que hpf::recommend sobre la misma señal), que sin transporte no escucha nada si nadie se lo pide y
// que el análisis vuelve con la sesión sin volver a escuchar.
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_events/juce_events.h>

#include <hpf/hpf.h>

#include <cmath>
#include <cstdio>
#include <random>

namespace {

constexpr double sr = 48000.0, pi = 3.14159265358979323846;
constexpr int blockSize = 512;
int failures = 0;

void check(bool ok, const char* what)   // texto en UTF-8, tal cual
{
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

struct Transport : juce::AudioPlayHead {
    bool playing = false;
    juce::int64 samples = 0;
    juce::Optional<PositionInfo> getPosition() const override
    {
        PositionInfo p;
        p.setIsPlaying(playing);
        p.setTimeInSamples(samples);
        p.setTimeInSeconds(static_cast<double>(samples) / sr);
        return p;
    }
};

// E2 y A2 alternados cada medio segundo sobre ruido grave por debajo de 25 Hz: hay algo que quitar
std::vector<float> makeTrack(double seconds)
{
    const auto n = static_cast<std::size_t>(seconds * sr);
    std::vector<float> x(n);
    std::mt19937 rng(5);
    std::normal_distribution<double> g(0.0, 1.0);
    const double a = std::exp(-2.0 * pi * 25.0 / sr);
    double lp1 = 0.0, lp2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double f0 = (static_cast<std::size_t>(static_cast<double>(i) / (0.5 * sr)) % 2 == 0) ? 82.41 : 110.0;
        double v = 0.0;
        for (int k = 1; k <= 10; ++k) v += std::sin(2.0 * pi * f0 * k * static_cast<double>(i) / sr) / k;
        lp1 = a * lp1 + (1.0 - a) * g(rng);
        lp2 = a * lp2 + (1.0 - a) * lp1;
        x[i] = static_cast<float>(0.2 * v + 3.0 * lp2);
    }
    return x;
}

juce::AudioProcessorParameter* param(juce::AudioPluginInstance& p, const juce::String& name)
{
    for (auto* q : p.getParameters())
        if (q->getName(64) == name) return q;
    return nullptr;
}

struct Settings {
    double cutoff = 0.0;
    int order = 0;
    bool active = false;
};

Settings read(juce::AudioPluginInstance& p)
{
    Settings s;
    if (auto* c = param(p, "Cutoff")) s.cutoff = c->getText(c->getValue(), 32).getDoubleValue();
    if (auto* o = param(p, "Slope")) s.order = o->getText(o->getValue(), 32).getIntValue() / 6;
    if (auto* a = param(p, "Filter")) s.active = a->getValue() >= 0.5f;
    return s;
}

// procesa la pista en bloques estéreo idénticos (una pista mono en un bus estéreo). En tiempo real
// cada bloque espera lo que dura, como en un DAW: si no, la cola del plugin (algo más de un segundo)
// se llena y se pierden muestras, que es lo que debe pasar si un host entrega audio sin reloj.
void play(juce::AudioPluginInstance& p, Transport* t, const std::vector<float>& x, bool realtime)
{
    juce::AudioBuffer<float> buf(2, blockSize);
    juce::MidiBuffer midi;
    for (std::size_t start = 0; start < x.size(); start += blockSize) {
        const int len = static_cast<int>(std::min<std::size_t>(blockSize, x.size() - start));
        buf.setSize(2, len, false, false, true);
        for (int ch = 0; ch < 2; ++ch) buf.copyFrom(ch, 0, x.data() + start, len);
        p.processBlock(buf, midi);
        if (t != nullptr) t->samples += len;
        if (realtime) juce::Thread::sleep(static_cast<int>(1000.0 * len / sr));
    }
}

// sigue llamando al plugin con silencio y atendiendo mensajes hasta que cambia el corte o pasa el tiempo
bool waitForApply(juce::AudioPluginInstance& p, Settings before, int maxMs)
{
    juce::AudioBuffer<float> silence(2, blockSize);
    juce::MidiBuffer midi;
    for (int waited = 0; waited < maxMs; waited += 20) {
        silence.clear();
        p.processBlock(silence, midi);
        juce::MessageManager::getInstance()->runDispatchLoopUntil(20);
        const auto now = read(p);
        if (std::abs(now.cutoff - before.cutoff) > 1e-6 || now.order != before.order || now.active != before.active) return true;
    }
    return false;
}

std::unique_ptr<juce::AudioPluginInstance> load(juce::AudioPluginFormatManager& fm, const juce::PluginDescription& d)
{
    juce::String error;
    auto p = fm.createPluginInstance(d, sr, blockSize, error);
    if (p == nullptr) std::printf("could not load: %s\n", error.toRawUTF8());
    return p;
}

void prepare(juce::AudioPluginInstance& p)
{
    p.enableAllBuses();
    p.setRateAndBufferSizeDetails(sr, blockSize);
    p.prepareToPlay(sr, blockSize);
}

}  // namespace

int main()
{
    juce::ScopedJuceInitialiser_GUI init;
    juce::AudioPluginFormatManager fm;
    juce::addDefaultFormatsToManager(fm);
    juce::OwnedArray<juce::PluginDescription> types;
    for (auto* format : fm.getFormats())
        if (format->getName() == "VST3") format->findAllTypesForFile(types, HPF_TEST_VST3);
    if (types.isEmpty()) {
        std::printf("FAIL: VST3 not found at %s\n", HPF_TEST_VST3);
        return 1;
    }

    const auto x = makeTrack(8.0);
    hpf::Channels mono(1, std::vector<double>(x.begin(), x.end()));
    const auto expected = hpf::recommend(mono, sr);
    const auto same = [&](const Settings& s) {
        return expected.cutoff_hz && std::abs(s.cutoff - *expected.cutoff_hz) < 0.05 && s.order == expected.order
               && s.active == expected.active;
    };
    std::printf("Core: cutoff %.1f Hz, %d dB/oct, filters %s\n", expected.cutoff_hz.value_or(0.0), 6 * expected.order,
                expected.active ? "yes" : "no");

    // 1. en un DAW: recién insertado escucha solo; al detenerse el transporte analiza y aplica
    Transport transport;
    juce::MemoryBlock session;
    {
        auto p = load(fm, *types[0]);
        if (p == nullptr) return 1;
        p->setPlayHead(&transport);
        prepare(*p);
        const auto before = read(*p);
        transport.playing = true;
        play(*p, &transport, x, true);
        transport.playing = false;
        check(waitForApply(*p, before, 20000), "with transport: nothing was applied when it stopped");
        const auto after = read(*p);
        std::printf("Transport: cutoff %.1f Hz, %d dB/oct, filters %s\n", after.cutoff, 6 * after.order, after.active ? "yes" : "no");
        check(same(after), "with transport: what was applied is not what the core recommends");
        p->getStateInformation(session);
        p->releaseResources();
    }

    // 2. render fuera de tiempo real, sin reproducir: escucha durante el render y analiza al terminar
    {
        auto p = load(fm, *types[0]);
        if (p == nullptr) return 1;
        Transport stopped;
        p->setPlayHead(&stopped);
        p->setNonRealtime(true);
        prepare(*p);
        const auto before = read(*p);
        play(*p, nullptr, x, false);
        p->setNonRealtime(false);
        check(waitForApply(*p, before, 20000), "render: nothing was applied when it ended");
        const auto after = read(*p);
        std::printf("Render: cutoff %.1f Hz, %d dB/oct, filters %s\n", after.cutoff, 6 * after.order, after.active ? "yes" : "no");
        check(same(after), "render: what was applied is not what the core recommends");
        p->releaseResources();
    }

    // 3. sin transporte (como la aplicación independiente) no escucha si nadie pulsa Escuchar
    {
        auto p = load(fm, *types[0]);
        if (p == nullptr) return 1;
        prepare(*p);
        const auto before = read(*p);
        play(*p, nullptr, x, false);
        check(!waitForApply(*p, before, 3000), "without transport: it applied something nobody asked for");
        p->releaseResources();
    }

    // 4. la sesión guardada trae el análisis y lo aplicado; al abrirla no vuelve a escuchar
    {
        auto p = load(fm, *types[0]);
        if (p == nullptr) return 1;
        Transport replay;
        p->setPlayHead(&replay);
        p->setStateInformation(session.getData(), static_cast<int>(session.getSize()));
        prepare(*p);
        const auto restored = read(*p);
        check(same(restored), "session: the parameters did not come back as saved");
        replay.playing = true;
        play(*p, &replay, x, false);
        replay.playing = false;
        check(!waitForApply(*p, restored, 3000), "session: it listened again and changed what was applied");
        p->releaseResources();
    }

    // 5. el host corre a 44.1 kHz y el render sale a 48 kHz: el host prepara el plugin de nuevo al
    // empezar el render y al terminarlo, como hace un DAW que renderiza a otra frecuencia; lo que se
    // escuchó en el render se analiza a 48 kHz y se aplica
    {
        auto p = load(fm, *types[0]);
        if (p == nullptr) return 1;
        Transport stopped;
        p->setPlayHead(&stopped);
        p->enableAllBuses();
        p->prepareToPlay(44100.0, blockSize);
        const auto before = read(*p);
        p->setNonRealtime(true);
        p->prepareToPlay(sr, blockSize);
        play(*p, nullptr, x, false);
        p->setNonRealtime(false);
        p->prepareToPlay(44100.0, blockSize);
        check(waitForApply(*p, before, 20000), "render at another rate: nothing was applied when it ended");
        const auto after = read(*p);
        std::printf("48 kHz render with the host at 44.1 kHz: cutoff %.1f Hz, %d dB/oct, filters %s\n", after.cutoff,
                    6 * after.order, after.active ? "yes" : "no");
        check(same(after), "render at another rate: what was applied is not what the core recommends at 48 kHz");
        p->releaseResources();
    }

    // 6. la salida es la cadena del núcleo: el filtro fijo de 20 Hz siempre, con el recomendado
    // apagado y encendido (sin transporte el plugin no escucha, así que no cambia nada solo)
    for (const bool on : {false, true}) {
        auto p = load(fm, *types[0]);
        if (p == nullptr) return 1;
        Transport stopped;
        p->setPlayHead(&stopped);
        hpf::Recommendation chain = expected;
        chain.active = on;
        if (on) {
            auto* c = param(*p, "Cutoff");
            auto* o = param(*p, "Slope");
            auto* a = param(*p, "Filter");
            c->setValueNotifyingHost(c->getValueForText(juce::String(*expected.cutoff_hz, 1)));
            o->setValueNotifyingHost(o->getValueForText(juce::String(6 * expected.order) + " dB/oct"));
            a->setValueNotifyingHost(1.0f);
            check(same(read(*p)), "chain: the core's parameters could not be set");
        }
        prepare(*p);
        std::vector<float> y;
        y.reserve(x.size());
        juce::AudioBuffer<float> buf(2, blockSize);
        juce::MidiBuffer midi;
        for (std::size_t start = 0; start < x.size(); start += blockSize) {
            const int len = static_cast<int>(std::min<std::size_t>(blockSize, x.size() - start));
            buf.setSize(2, len, false, false, true);
            for (int ch = 0; ch < 2; ++ch) buf.copyFrom(ch, 0, x.data() + start, len);
            p->processBlock(buf, midi);
            y.insert(y.end(), buf.getReadPointer(0), buf.getReadPointer(0) + len);
        }
        auto want = mono;
        hpf::apply_recommendation(want, sr, chain);
        double e = 0.0, w = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            e += (y[i] - want[0][i]) * (y[i] - want[0][i]);
            w += want[0][i] * want[0][i];
        }
        const double residual = 10.0 * std::log10(e / w + 1e-30);
        std::printf("Chain with the recommended filter %s: residual %.1f dB against the core\n", on ? "on" : "off", residual);
        check(residual < -80.0, on ? "chain: with the recommended filter on the output is not the core's"
                                   : "chain: with the recommended filter off the fixed 20 Hz filter is missing");
        p->releaseResources();
    }

    std::printf("%s.\n", failures ? "With failures" : "Passed");
    return failures ? 1 : 0;
}
