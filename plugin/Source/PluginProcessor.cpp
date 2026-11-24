#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <hpf/hpf.h>

namespace {
const juce::Identifier analysisTag{"Analisis"};
const juce::Identifier autoApplyTag{"autoApply"};
}

HPFRecommenderProcessor::HPFRecommenderProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "HPFRecommender", createLayout()),
      analyser([this] { triggerAsyncUpdate(); })   // al terminar un análisis, aplicar en el hilo de mensajes
{
    cutoffParam = parameters.getRawParameterValue("cutoff");
    orderParam = parameters.getRawParameterValue("slope");
    activeParam = parameters.getRawParameterValue("active");
}

juce::AudioProcessorValueTreeState::ParameterLayout HPFRecommenderProcessor::createLayout()
{
    juce::NormalisableRange<float> range(minCutoff, maxCutoff);
    range.setSkewForCentre(69.3f);   // el punto medio del control cae en el centro geométrico de 16-300 Hz
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"cutoff", 1}, "Cutoff", range, 20.0f,
        juce::AudioParameterFloatAttributes().withLabel("Hz").withStringFromValueFunction(
            [](float value, int) { return juce::String(value, value < 100.0f ? 1 : 0); })));
    juce::StringArray slopes;
    for (int order = 1; order <= hpfplugin::maxOrder; ++order) slopes.add(juce::String(6 * order) + " dB/oct");
    layout.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"slope", 1}, "Slope", slopes, 1));
    layout.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"active", 1}, "Filter", false));
    return layout;
}

bool HPFRecommenderProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    return layouts.getMainInputChannelSet() == out
           && (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo());
}

void HPFRecommenderProcessor::prepareToPlay(double newSampleRate, int)
{
    sampleRate = newSampleRate;
    cutoff.reset(sampleRate, 0.05);
    cutoff.setCurrentAndTargetValue(juce::jlimit<double>(minCutoff, maxCutoff, cutoffParam->load()));
    wet.reset(sampleRate, 0.02);
    wet.setCurrentAndTargetValue(activeParam->load() >= 0.5f ? 1.0f : 0.0f);
    designedCutoff = 0.0;   // obliga a diseñar el filtro en el primer bloque
    designedOrder = 0;
    filter.reset();
    base.reset();
    base.setCoefficients(hpfplugin::butterworthHighPass(hpf::base_filter_hz, hpf::base_filter_order, sampleRate));
    dry.setSize(hpfplugin::HighPass::maxChannels, chunk);
    analyser.prepare(sampleRate, juce::jmax(1, getTotalNumInputChannels()));
    // Recién insertado en un DAW queda escuchando: basta reproducir (o hacer un render) una vez.
    // En la aplicación independiente no, porque ahí no hay transporte que marque el final.
    if (!autoListenArmed) {
        autoListenArmed = true;
        if (wrapperType != wrapperType_Standalone && !analyser.getResult()) analyser.startListening();
    }
}

void HPFRecommenderProcessor::startListeningManually()
{
    manualListen = true;
    analyser.startListening();
}

bool HPFRecommenderProcessor::getAutoApply() const
{
    return parameters.state.getProperty(autoApplyTag, true);
}

void HPFRecommenderProcessor::setAutoApply(bool shouldApply)
{
    parameters.state.setProperty(autoApplyTag, shouldApply, nullptr);
}

void HPFRecommenderProcessor::handleAsyncUpdate()
{
    if (getAutoApply()) applyRecommendation();
}

void HPFRecommenderProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(getTotalNumInputChannels(), hpfplugin::HighPass::maxChannels);
    for (int ch = getTotalNumInputChannels(); ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear(ch, 0, numSamples);

    // El análisis escucha la señal original, antes del filtro: mientras el DAW reproduce o hace un
    // render (fuera de tiempo real se espera lugar en la cola). Sin transporte, solo si el usuario
    // pulsó Escuchar.
    if (analyser.isListening()) {
        const bool offline = isNonRealtime();
        bool playing = manualListen.load();
        if (auto* playHead = getPlayHead())
            if (const auto position = playHead->getPosition())
                playing = position->getIsPlaying();
        if (offline || playing) analyser.push(buffer, offline);
    }

    cutoff.setTargetValue(juce::jlimit<double>(minCutoff, maxCutoff, cutoffParam->load()));
    wet.setTargetValue(activeParam->load() >= 0.5f ? 1.0f : 0.0f);
    const int order = 1 + juce::jlimit(0, hpfplugin::maxOrder - 1, juce::roundToInt(orderParam->load()));

    // El filtro fijo de 20 Hz va siempre, delante de todo; el análisis ya escuchó la señal original y el
    // núcleo aplica el mismo fijo antes de analizar.
    {
        float* all[hpfplugin::HighPass::maxChannels] = {};
        for (int ch = 0; ch < numChannels; ++ch) all[ch] = buffer.getWritePointer(ch);
        base.process(all, numChannels, numSamples);
    }

    // el filtro recomendado corre siempre, para que su estado esté listo; al activarlo o desactivarlo
    // se mezcla con la salida del fijo durante 20 ms en lugar de conmutar
    for (int start = 0; start < numSamples; start += chunk) {
        const int len = juce::jmin(chunk, numSamples - start);
        const double fc = cutoff.skip(len);
        if (!juce::exactlyEqual(fc, designedCutoff) || order != designedOrder) {
            filter.setCoefficients(hpfplugin::butterworthHighPass(fc, order, sampleRate));
            designedCutoff = fc;
            designedOrder = order;
        }
        float* channels[hpfplugin::HighPass::maxChannels] = {};
        for (int ch = 0; ch < numChannels; ++ch) {
            dry.copyFrom(ch, 0, buffer, ch, start, len);
            channels[ch] = buffer.getWritePointer(ch, start);
        }
        filter.process(channels, numChannels, len);
        if (wet.isSmoothing()) {
            for (int i = 0; i < len; ++i) {
                const float g = wet.getNextValue();
                for (int ch = 0; ch < numChannels; ++ch)
                    channels[ch][i] = g * channels[ch][i] + (1.0f - g) * dry.getSample(ch, i);
            }
        } else if (wet.getTargetValue() < 0.5f) {
            for (int ch = 0; ch < numChannels; ++ch) buffer.copyFrom(ch, start, dry, ch, 0, len);
        }
    }
}

void HPFRecommenderProcessor::setParameterValue(const juce::String& id, float plainValue)
{
    if (auto* parameter = parameters.getParameter(id)) {
        parameter->beginChangeGesture();
        parameter->setValueNotifyingHost(parameter->convertTo0to1(plainValue));
        parameter->endChangeGesture();
    }
}

void HPFRecommenderProcessor::applyRecommendation()
{
    const auto result = analyser.getResult();
    if (!result || !result->cutoffHz) return;
    setParameterValue("cutoff", static_cast<float>(juce::jlimit<double>(minCutoff, maxCutoff, *result->cutoffHz)));
    setParameterValue("slope", static_cast<float>(juce::jlimit(1, hpfplugin::maxOrder, result->order) - 1));
    setParameterValue("active", result->active ? 1.0f : 0.0f);   // apagado si no hay nada medible que quitar
}

void HPFRecommenderProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    state.removeChild(state.getChildWithName(analysisTag), nullptr);
    if (const auto result = analyser.getResult()) {
        juce::ValueTree analysis(analysisTag);
        analysis.setProperty("cutoff", result->cutoffHz ? *result->cutoffHz : -1.0, nullptr);
        analysis.setProperty("order", result->order, nullptr);
        analysis.setProperty("active", result->active, nullptr);
        analysis.setProperty("summary", result->summary, nullptr);
        analysis.setProperty("json", result->json, nullptr);
        state.appendChild(analysis, nullptr);
    }
    if (const auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void HPFRecommenderProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    const auto xml = getXmlFromBinary(data, sizeInBytes);
    if (xml == nullptr || !xml->hasTagName(parameters.state.getType())) return;
    auto state = juce::ValueTree::fromXml(*xml);
    const auto analysis = state.getChildWithName(analysisTag);
    if (analysis.isValid()) {
        hpfplugin::Analysis restored;
        const double fc = analysis.getProperty("cutoff", -1.0);
        if (fc > 0.0) restored.cutoffHz = fc;
        restored.order = analysis.getProperty("order", 0);
        restored.active = analysis.getProperty("active", false);
        restored.summary = analysis.getProperty("summary").toString();
        restored.json = analysis.getProperty("json").toString();
        analyser.restoreResult(std::move(restored));
        state.removeChild(analysis, nullptr);
    }
    parameters.replaceState(state);
}

juce::AudioProcessorEditor* HPFRecommenderProcessor::createEditor()
{
    return new HPFRecommenderEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new HPFRecommenderProcessor();
}
