// Interfaz: escuchar y analizar, leer la recomendación, aplicarla y ajustar el filtro a mano.
#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include "PluginProcessor.h"

class HPFRecommenderEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit HPFRecommenderEditor(HPFRecommenderProcessor& processorToEdit);
    ~HPFRecommenderEditor() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void timerCallback() override;

    HPFRecommenderProcessor& owner;
    juce::Label title, status, cutoffLabel, orderLabel;
    juce::TextButton listenButton, applyButton;
    juce::TextEditor report;
    juce::Slider cutoffSlider;
    juce::ComboBox orderBox;
    juce::ToggleButton activeButton, autoApplyButton;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> cutoffAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> orderAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> activeAttachment;
    int shownVersion = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HPFRecommenderEditor)
};
