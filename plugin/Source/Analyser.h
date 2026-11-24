// Escucha la pista mientras suena y, al terminar, le pide la recomendación al núcleo.
//
// El hilo de audio solo copia muestras a una cola sin bloqueo (juce::AbstractFifo) cuya memoria se
// reserva en prepare(). El hilo de análisis vacía la cola, acumula la pista y, cuando se le pide o
// cuando deja de llegar audio (se detuvo el transporte o terminó un render), llama a
// hpf::recommend con la pista completa. Nada de eso ocurre en el hilo de audio.
#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace hpfplugin {

struct Analysis {
    std::optional<double> cutoffHz;   // vacío si no hay filtro que recomendar
    int order = 0;
    bool active = false;              // si conviene filtrar: hay algo medible que quitar
    juce::String summary;             // texto para la interfaz
    juce::String json;                // reporte completo del núcleo
};

class Analyser : private juce::Thread {
public:
    enum class Stage { idle, listening, analysing, done, failed };
    static constexpr double maxSeconds = 600.0;
    static constexpr double minSeconds = 3.0;    // menos que esto no se analiza solo
    static constexpr int idleMs = 1000;          // sin audio nuevo durante esto: se analiza

    // onFinished se llama desde el hilo de análisis cada vez que termina un análisis nuevo
    explicit Analyser(std::function<void()> onFinished = {});
    ~Analyser() override;

    // fuera del hilo de audio. Si el host cambia de formato mientras escucha (empieza o termina un
    // render a otra frecuencia), lo escuchado se analiza a su propia frecuencia si alcanza (minSeconds);
    // si no alcanza, se descarta y se sigue escuchando en el formato nuevo.
    void prepare(double sampleRate, int numChannels);
    // hilo de audio. Con mayWait (render fuera de tiempo real) espera a que haya lugar en la cola
    // en vez de perder muestras: ahí el bloque puede tardar.
    void push(const juce::AudioBuffer<float>& buffer, bool mayWait = false) noexcept;
    bool isListening() const noexcept { return listening.load(std::memory_order_acquire); }

    void startListening();
    void stopAndAnalyse();

    Stage getStage() const noexcept { return stage.load(); }
    double getCapturedSeconds() const noexcept;
    std::optional<Analysis> getResult() const;
    int getResultVersion() const noexcept { return resultVersion.load(); }
    void restoreResult(Analysis analysis);

private:
    void run() override;
    void drainLocked();
    void analyse();

    juce::AbstractFifo fifo{1};
    std::vector<std::vector<float>> queue;      // memoria de la cola, reservada en prepare()
    std::vector<std::vector<float>> captured;   // lo toca el hilo de análisis, con configMutex
    bool identicalChannels = true;              // mientras L = R se guarda un solo canal
    std::function<void()> onFinished;
    std::mutex configMutex;
    mutable std::mutex resultMutex;
    std::optional<Analysis> result;
    int channels = 0;
    std::atomic<double> rate{0.0};          // la del host, desde el último prepare()
    std::atomic<double> captureRate{0.0};   // la de lo capturado, que el host puede haber cambiado
    std::atomic<bool> listening{false}, analyseRequested{false};
    std::atomic<Stage> stage{Stage::idle};
    std::atomic<long long> capturedSamples{0}, droppedSamples{0};
    std::atomic<int> resultVersion{0};
};

}  // namespace hpfplugin
