cmake_minimum_required(VERSION 3.22)
project(HPFRecommenderPlugin VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "macOS mínimo")

set(JUCE_DIR "/Applications/JUCE-9.0.6" CACHE PATH "Carpeta de JUCE (probado con 8.0.12 y 9.0.6)")
set(HPF_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp" CACHE PATH "C++ core of the recommender (../cpp)")

option(HPF_COPY_PLUGIN "Install the built plugins in the user's plug-in folders" ON)
if(APPLE)
    set(HPF_FORMATS AU VST3 Standalone)
else()
    set(HPF_FORMATS VST3 Standalone)
endif()

add_subdirectory("${JUCE_DIR}" JUCE)
add_subdirectory("${HPF_CORE_DIR}" hpf_core)   # también compila sus pruebas, que ctest corre
# el núcleo estático va dentro de un módulo compartido (VST3 en Linux): necesita código independiente de la posición
set_target_properties(hpf PROPERTIES POSITION_INDEPENDENT_CODE ON)

juce_add_plugin(HPFRecommender
    COMPANY_NAME "Atonal"
    BUNDLE_ID com.atonal.hpfrecommender
    PLUGIN_MANUFACTURER_CODE Atnl
    PLUGIN_CODE Hpfr
    FORMATS ${HPF_FORMATS}
    PRODUCT_NAME "HPF!"
    VERSION 0.3.0
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD ${HPF_COPY_PLUGIN}
    AU_MAIN_TYPE kAudioUnitType_Effect
    VST3_CATEGORIES Fx Filter)

target_sources(HPFRecommender PRIVATE
    Source/Analyser.cpp
    Source/PluginEditor.cpp
    Source/PluginProcessor.cpp)
target_include_directories(HPFRecommender PRIVATE Source)
target_compile_definitions(HPFRecommender PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(HPFRecommender
    PRIVATE hpf juce::juce_audio_utils
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags juce::juce_recommended_warning_flags)

# Pruebas del filtro contra el núcleo, sin JUCE
add_executable(hpf_plugin_tests tests/test_highpass.cpp)
target_include_directories(hpf_plugin_tests PRIVATE Source "${HPF_CORE_DIR}/src")
target_link_libraries(hpf_plugin_tests PRIVATE hpf)

# Camino de escucha y análisis del plugin, sin DAW
juce_add_console_app(hpf_analyser_test PRODUCT_NAME "hpf_analyser_test")
target_sources(hpf_analyser_test PRIVATE tests/test_analyser.cpp Source/Analyser.cpp)
target_include_directories(hpf_analyser_test PRIVATE Source)
target_compile_definitions(hpf_analyser_test PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(hpf_analyser_test
    PRIVATE hpf juce::juce_audio_basics juce::juce_core
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)

# El plugin compilado, cargado como VST3 en un host mínimo con transporte simulado
juce_add_console_app(hpf_host_test PRODUCT_NAME "hpf_host_test")
target_sources(hpf_host_test PRIVATE tests/test_host.cpp)
target_compile_definitions(hpf_host_test PRIVATE
    JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_PLUGINHOST_VST3=1 JUCE_PLUGINHOST_AU=0 JUCE_MODAL_LOOPS_PERMITTED=1
    HPF_TEST_VST3="$<TARGET_GENEX_EVAL:HPFRecommender_VST3,$<TARGET_PROPERTY:HPFRecommender_VST3,JUCE_PLUGIN_ARTEFACT_FILE>>")
target_link_libraries(hpf_host_test
    PRIVATE hpf juce::juce_audio_processors juce::juce_events
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
add_dependencies(hpf_host_test HPFRecommender_VST3)

enable_testing()
add_test(NAME highpass COMMAND hpf_plugin_tests)
add_test(NAME analyser COMMAND hpf_analyser_test)
add_test(NAME host COMMAND hpf_host_test)
