// Recorre el camino de escucha del plugin sin un DAW: entrega una pista al analizador por bloques,
// como lo haría el hilo de audio, y comprueba que el reporte es idéntico al de hpf::recommend con la
// misma señal y que no se perdió ninguna muestra. También comprueba que el análisis empieza solo
// cuando deja de llegar audio, que una pista con L = R se analiza como mono y que se avisa al terminar.
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <hpf/hpf.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <random>

#include "Analyser.h"

namespace {

constexpr double sr = 48000.0, pi = 3.14159265358979323846;
constexpr int seconds = 8, block = 4800;
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// bajo en E1 con armónicos y algo de ruido; el canal derecho con la ganancia dada
juce::AudioBuffer<float> bass(float rightGain)
{
    const int n = seconds * static_cast<int>(sr);
    juce::AudioBuffer<float> track(2, n);
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 0.002f);
    for (int i = 0; i < n; ++i) {
        double v = 0.0;
        for (int k = 1; k <= 8; ++k) v += std::sin(2.0 * pi * 41.2 * k * i / sr) / k;
        const float s = static_cast<float>(0.2 * v) + noise(rng);
        track.setSample(0, i, s);
        track.setSample(1, i, rightGain * s);
    }
    return track;
}

void feed(hpfplugin::Analyser& analyser, juce::AudioBuffer<float>& track)
{
    const int n = track.getNumSamples();
    for (int start = 0; start < n; start += block) {
        juce::AudioBuffer<float> chunk(track.getArrayOfWritePointers(), 2, start, juce::jmin(block, n - start));
        analyser.push(chunk);
        juce::Thread::sleep(10);
    }
}

bool waitDone(const hpfplugin::Analyser& analyser, int maxMs)
{
    for (int waited = 0; waited < maxMs; waited += 50) {
        const auto stage = analyser.getStage();
        if (stage == hpfplugin::Analyser::Stage::done || stage == hpfplugin::Analyser::Stage::failed) return true;
        juce::Thread::sleep(50);
    }
    return false;
}

hpf::Channels toChannels(const juce::AudioBuffer<float>& track, int numChannels)
{
    hpf::Channels x(static_cast<std::size_t>(numChannels), std::vector<double>(static_cast<std::size_t>(track.getNumSamples())));
    for (int ch = 0; ch < numChannels; ++ch)
        for (int i = 0; i < track.getNumSamples(); ++i)
            x[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)] = track.getSample(ch, i);
    return x;
}

}  // namespace

int main()
{
    // 1. estéreo distinto (derecho a -6 dB), detenido a mano: el mismo reporte que el núcleo
    std::atomic<int> finished{0};
    hpfplugin::Analyser analyser([&finished] { ++finished; });
    analyser.prepare(sr, 2);
    auto stereo = bass(0.5f);
    analyser.startListening();
    feed(analyser, stereo);
    analyser.stopAndAnalyse();
    check(waitDone(analyser, 60000) && analyser.getStage() == hpfplugin::Analyser::Stage::done, "the analysis did not finish");
    const auto direct = hpf::recommend(toChannels(stereo, 2), sr);
    auto result = analyser.getResult();
    check(result && result->json == juce::String::fromUTF8(hpf::to_json(direct).c_str()),
          "the plugin's report differs from hpf::recommend's");
    check(result && result->active == direct.active && result->order == direct.order,
          "the plugin's decision (filter, slope) differs from the core's");
    check(result && !result->summary.contains("se perdieron"), "samples were lost while listening");
    check(finished.load() == 1, "no notification when the analysis finished");
    if (result)
        std::printf("Stereo: cutoff %.1f Hz, order %d, filters %s\n%s\n", result->cutoffHz.value_or(0.0), result->order,
                    result->active ? "yes" : "no", result->summary.toRawUTF8());

    // 2. L = R y sin pulsar nada: al dejar de llegar audio se analiza solo, como mono
    auto twin = bass(1.0f);
    analyser.startListening();
    feed(analyser, twin);
    check(waitDone(analyser, 60000), "the analysis did not start by itself when audio stopped arriving");
    const auto mono = hpf::recommend(toChannels(twin, 1), sr);
    result = analyser.getResult();
    check(result && result->json == juce::String::fromUTF8(hpf::to_json(mono).c_str()),
          "with L = R the report is not that of the mono track");
    check(finished.load() == 2, "no notification when the second analysis finished");

    // 3. el host corre a 44.1 kHz y el render sale a 48 kHz: el host vuelve a preparar el plugin al
    // empezar el render (todavía no se escuchó nada: sigue escuchando a 48 kHz) y al terminar, antes
    // del segundo sin audio (lo escuchado se analiza enseguida, a la frecuencia a la que se escuchó)
    hpfplugin::Analyser other;
    other.prepare(44100.0, 2);
    other.startListening();
    other.prepare(sr, 2);
    check(other.isListening(), "stopped listening when the host changed rate before hearing anything");
    feed(other, twin);
    other.prepare(44100.0, 2);
    check(waitDone(other, 60000), "with a render at another rate there was no analysis");
    result = other.getResult();
    check(result && result->json == juce::String::fromUTF8(hpf::to_json(mono).c_str()),
          "with a render at another rate the report is not that of the 48 kHz track");

    std::printf("%s.\n", failures ? "With failures" : "Passed");
    return failures ? 1 : 0;
}
