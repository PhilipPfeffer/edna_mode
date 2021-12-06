#include "mean_embeddings.h"
#include "micro_features_model.h"

#ifdef TINY_CONV_OLD
const char *unknown_label = "UNKNOWN USER";
const char *mean_embeddings_labels[num_labels] = {"phil","arden","greg"};
const int8_t mean_embeddings[num_labels][embedding_size] = {
{27,-74,67,-83,-55,29,46,41,43,-23,-10,-19,13,-66,-53,-73,-59,-66,-24,-51},
{-36,11,-88,21,-5,-61,-68,-69,-52,-11,-31,-28,-64,6,-9,1,14,10,-17,2},
{-43,-8,-36,-12,-11,-31,-38,-35,-45,-26,-21,-18,-20,-8,-12,-4,-19,-13,-22,-18},
};
const float thresholds[num_labels] = {366.71923865540515,253.78731252763603,302.43015722642474};
#endif

#ifdef MOBILENET
const char *unknown_label = "UNKNOWN USER";
const char *mean_embeddings_labels[num_labels] = {"greg","phil","faith","arden"};
const int8_t mean_embeddings[num_labels][embedding_size] = {
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
{61,-118,-56,43,-80,-128,22,87,-18,-6,-109,-77,-109,44,70,67,-1,51,-55,49,-58,-53,106,50,32,9,-120,-75,-78,-5,89,64,-49,-95,-35,-30,-123,52,51,-65,-87,-32,127,-25,31,-97,99,-14,2,-94},
};
const float thresholds[num_labels] = {0.0,1.7320508075688772,1.7320508075688772,1.7320508075688772};
#endif
