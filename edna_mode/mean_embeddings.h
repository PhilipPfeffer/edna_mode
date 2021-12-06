// Labels and their mean embeddings.
#include <stdint.h> 

#define num_labels 3
#define embedding_size 50
extern const char *unknown_label;
extern const char *mean_embeddings_labels[num_labels];
extern const float thresholds[num_labels];
extern const int8_t mean_embeddings[num_labels][embedding_size];
