// Para compilar con RealtimeSanitizer (Clang de LLVM, -fsanitize=realtime): rediseñar el filtro y
// procesar un bloque dentro de una función de tiempo real no debe reservar memoria ni bloquear.
#include <vector>

#include "HighPass.h"

namespace {

void audioCallback(hpfplugin::HighPass& hp, float* const* data, int numSamples, double fc, int order)
    [[clang::nonblocking]]
{
    hp.setCoefficients(hpfplugin::butterworthHighPass(fc, order, 48000.0));
    hp.process(data, 2, numSamples);
}

}  // namespace

int main()
{
    std::vector<float> left(512, 0.1f), right(512, -0.1f);
    float* data[2] = {left.data(), right.data()};
    hpfplugin::HighPass hp;
    for (int i = 0; i < 1000; ++i)
        audioCallback(hp, data, 512, 16.0 + 0.28 * i, 1 + i % hpfplugin::maxOrder);
    return 0;
}
