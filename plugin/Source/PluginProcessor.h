// Procesador del plugin: escucha la pista para el análisis y aplica el pasa-altos.
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "Analyser.h"
#include "HighPass.h"

class HPFRecommenderProcessor : public juce::AudioProcessor, private juce::AsyncUpdater {
public:
    static constexpr float minCutoff = 16.0f, maxCutoff = 300.0f;

    HPFRecommenderProcessor();

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Pasa a los parámetros lo que decidió el análisis: corte, pendiente y si conviene filtrar.
    void applyRecommendation();
    // Escuchar a pedido del usuario: sin transporte (aplicación independiente) se escucha todo.
    void startListeningManually();
    // Aplicar solo al terminar cada análisis (se guarda con la sesión; por omisión, sí).
    bool getAutoApply() const;
    void setAutoApply(bool shouldApply);

    juce::AudioProcessorValueTreeState parameters;
    hpfplugin::Analyser analyser;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void handleAsyncUpdate() override;
    void setParameterValue(const juce::String& id, float plainValue);

    static constexpr int chunk = 32;   // muestras entre recálculos del filtro mientras se suaviza el corte
    std::atomic<float>* cutoffParam = nullptr;
    std::atomic<float>* orderParam = nullptr;
    std::atomic<float>* activeParam = nullptr;
    hpfplugin::HighPass filter;
    hpfplugin::HighPass base;   // filtro fijo de 20 Hz, siempre puesto, delante del recomendado
    juce::SmoothedValue<double, juce::ValueSmoothingTypes::Multiplicative> cutoff;
    juce::SmoothedValue<float> wet;
    juce::AudioBuffer<float> dry;
    double sampleRate = 48000.0;
    double designedCutoff = 0.0;
    int designedOrder = 0;
    bool autoListenArmed = false;              // la primera preparación deja escuchando la pista
    std::atomic<bool> manualListen{false};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HPFRecommenderProcessor)
};
