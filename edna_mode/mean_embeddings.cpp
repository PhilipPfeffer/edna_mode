#include "mean_embeddings.h"
#include "micro_features_model.h"
#include <stdint.h> 

#ifdef TINY_CONV_OLD
const char *unknown_label = "UNKNOWN USER";
const char *mean_embeddings_labels[num_labels] = {"greg","phil","arden"};
const float mean_embeddings[num_labels][embedding_size] = {
{0.398980632,-0.10477792699999999,0.143777033,-0.01195924,-0.118035815,0.092800611,-0.15955755300000002,0.058616931,-0.00229704,-0.09940227199999999,-0.183736701,-0.027734118000000002,0.24388453499999999,-0.132438034,-0.0039653360000000025,0.029610986000000006,0.22282334099999995,-0.15006411999999997,-0.238564915,-0.09947023599999999},
{0.32456213079999996,-0.073876566,0.147784891,0.09949194100000001,-0.062333950799999996,-0.0764747218,-0.0626560024,0.26559875999999993,0.21086155399999998,-0.22893759130000002,-0.13368786700000004,-0.08610390500000001,0.0405935379,-0.02505493146799998,-0.19722486899999997,0.061154859799999996,0.085906533,-0.182720522731,-0.0811424068,-0.11926247100000001},
{0.4753022692,-0.171725214372,0.26319405358000003,0.2144471845639,-0.09594505057,-0.10877624290000001,-0.1944047109,0.1840269316,0.1118532861,-0.2822554852,-0.2393786526,-0.205492004234,0.062025708449999994,-0.19858635500000002,-0.1649954691,-0.031049315100000002,0.20471430709,-0.2319920385,-0.155177079086,-0.20635657251},
};
const float thresholds[num_labels] = {1.8937472908022708,1.8160229015105065,1.125747437100794};
#endif

#ifdef MOBILENET
const char *unknown_label = "UNKNOWN USER";
const char *mean_embeddings_labels[num_labels] = {"greg","phil","arden"};
const int8_t mean_embeddings[num_labels][embedding_size] = {
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
};
const float thresholds[num_labels] = {0.0,1.7320508075688772,1.7320508075688772};
#endif
