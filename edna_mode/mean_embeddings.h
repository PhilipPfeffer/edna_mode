// Labels and their mean embeddings.
#include "micro_features_model.h"

#ifdef TINY_CONV_OLD
#define num_labels 3
#define embedding_size 20
#endif
#ifdef MOBILENET
#define num_labels 3
#define embedding_size 50
#endif
extern const char *unknown_label;
extern const char *mean_embeddings_labels[num_labels];
extern const float thresholds[num_labels];
extern const int8_t mean_embeddings[num_labels][embedding_size];
