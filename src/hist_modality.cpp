#include "hist_modality.hpp"

namespace mb3t {
namespace hm{
    
    void HistModality::ssf_init() {
        ssf_lookup_f_.resize(ssf_length);
        ssf_lookup_b_.resize(ssf_length);
        for (int i = 0; i < ssf_length; ++i) {
            float x = static_cast<float>(i) - static_cast<float>(ssf_length - 1) / 2.0f;
            if (ssf_slope == 0.0f) {
                // why not (x < 0.0f? -1 : 1) ? answer: we want x==0.0f to be jsut 0.5f
                ssf_lookup_f_[i] = 0.5f - ssf_amptitue * ((0.0f < x) - (x < 0.0f)); 
            } else {
                ssf_lookup_f_[i] = 0.5f - ssf_amptitue * std::tanh(x / (2.0f * ssf_slope));
            }
            ssf_lookup_b_[i] = 1.0f - ssf_lookup_f_[i];
        }
    }

    void HistModality::hist_init() {
        hist_n_bins_square_ = hist_n_bins * hist_n_bins;
        hist_n_bins_cube_ = hist_n_bins_square_ * hist_n_bins;
        hist_temp_f_.resize(hist_n_bins_cube_);
        hist_temp_b_.resize(hist_n_bins_cube_);
        hist_f_.resize(hist_n_bins_cube_);
        hist_b_.resize(hist_n_bins_cube_);
    }

}
}