#include "PluginEditor.h"

namespace {

juce::String utf8(const char* text) { return juce::String::fromUTF8(text); }

const char* const placeholder =
    "Play the whole track once (or run a render). When it stops, the analysis starts by itself "
    "and the recommendation appears here.";

}  // namespace

HPFRecommenderEditor::HPFRecommenderEditor(HPFRecommenderProcessor& processorToEdit)
    : AudioProcessorEditor(processorToEdit), owner(processorToEdit)
{
    title.setText("HPF!", juce::dontSendNotification);
    title.setFont(juce::Font(juce::FontOptions(20.0f, juce::Font::bold)));
    addAndMakeVisible(title);

    status.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(status);

    listenButton.setButtonText("Listen");
    listenButton.onClick = [this] {
        if (owner.analyser.getStage() == hpfplugin::Analyser::Stage::listening)
            owner.analyser.stopAndAnalyse();
        else
            owner.startListeningManually();
    };
    addAndMakeVisible(listenButton);

    applyButton.setButtonText("Apply");
    applyButton.onClick = [this] { owner.applyRecommendation(); };
    addAndMakeVisible(applyButton);

    report.setMultiLine(true);
    report.setReadOnly(true);
    report.setCaretVisible(false);
    report.setScrollbarsShown(true);
    addAndMakeVisible(report);

    cutoffLabel.setText("Cutoff", juce::dontSendNotification);
    orderLabel.setText("Slope", juce::dontSendNotification);
    addAndMakeVisible(cutoffLabel);
    addAndMakeVisible(orderLabel);

    cutoffSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    cutoffSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 80, 22);
    cutoffSlider.setTextValueSuffix(" Hz");
    addAndMakeVisible(cutoffSlider);

    for (int order = 1; order <= hpfplugin::maxOrder; ++order) orderBox.addItem(juce::String(6 * order) + " dB/oct", order);
    addAndMakeVisible(orderBox);

    activeButton.setButtonText("Filter on");
    addAndMakeVisible(activeButton);

    autoApplyButton.setButtonText("Apply when done");
    autoApplyButton.setToggleState(owner.getAutoApply(), juce::dontSendNotification);
    autoApplyButton.onClick = [this] { owner.setAutoApply(autoApplyButton.getToggleState()); };
    addAndMakeVisible(autoApplyButton);

    using APVTS = juce::AudioProcessorValueTreeState;
    cutoffAttachment = std::make_unique<APVTS::SliderAttachment>(owner.parameters, "cutoff", cutoffSlider);
    orderAttachment = std::make_unique<APVTS::ComboBoxAttachment>(owner.parameters, "slope", orderBox);
    activeAttachment = std::make_unique<APVTS::ButtonAttachment>(owner.parameters, "active", activeButton);

    setSize(560, 400);
    timerCallback();
    startTimerHz(10);
}

HPFRecommenderEditor::~HPFRecommenderEditor()
{
    stopTimer();
}

void HPFRecommenderEditor::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void HPFRecommenderEditor::resized()
{
    auto area = getLocalBounds().reduced(14);
    title.setBounds(area.removeFromTop(30));
    area.removeFromTop(6);

    auto row = area.removeFromTop(30);
    listenButton.setBounds(row.removeFromLeft(170));
    row.removeFromLeft(10);
    status.setBounds(row);
    area.removeFromTop(10);

    auto controls = area.removeFromBottom(30);
    activeButton.setBounds(controls.removeFromRight(120));
    controls.removeFromRight(10);
    orderBox.setBounds(controls.removeFromRight(110));
    orderLabel.setBounds(controls.removeFromRight(75));
    cutoffLabel.setBounds(controls.removeFromLeft(50));
    cutoffSlider.setBounds(controls);
    area.removeFromBottom(10);

    auto applyRow = area.removeFromBottom(30);
    applyButton.setBounds(applyRow.removeFromRight(120));
    applyRow.removeFromRight(10);
    autoApplyButton.setBounds(applyRow.removeFromRight(170));
    area.removeFromBottom(8);
    report.setBounds(area);
}

void HPFRecommenderEditor::timerCallback()
{
    using Stage = hpfplugin::Analyser::Stage;
    const auto stage = owner.analyser.getStage();
    switch (stage) {
        case Stage::idle:
            status.setText("Ready to listen.", juce::dontSendNotification);
            break;
        case Stage::listening:
            status.setText("Listening: " + juce::String(owner.analyser.getCapturedSeconds(), 1)
                               + " s. It analyses by itself when the track stops.",
                           juce::dontSendNotification);
            break;
        case Stage::analysing:
            status.setText("Analysing...", juce::dontSendNotification);
            break;
        case Stage::done:
            status.setText(utf8("Recommendation ready."), juce::dontSendNotification);
            break;
        case Stage::failed:
            status.setText(utf8("The analysis did not finish."), juce::dontSendNotification);
            break;
    }
    listenButton.setButtonText(stage == Stage::listening ? "Analyse now" : stage == Stage::idle ? "Listen" : "Listen again");
    autoApplyButton.setToggleState(owner.getAutoApply(), juce::dontSendNotification);
    listenButton.setEnabled(stage != Stage::analysing);

    const int version = owner.analyser.getResultVersion();
    if (version != shownVersion) {
        shownVersion = version;
        const auto result = owner.analyser.getResult();
        report.setText(result ? result->summary : utf8(placeholder), false);
        applyButton.setEnabled(result.has_value() && result->cutoffHz.has_value());
    }
}
