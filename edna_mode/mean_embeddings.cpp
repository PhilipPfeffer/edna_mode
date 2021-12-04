#include "mean_embeddings.h"

const char *unknown_label = "UNKNOWN USER";
const char *mean_embeddings_labels[num_labels] = {"phil", "arden", "greg"};
const float mean_embeddings[num_labels][embedding_size] = {
{-0.004663365,-0.018474274999999995,-0.013126533000000001,0.043056192,0.029341207999999997,-0.039216233,-0.056548659999999994,0.036362676999999996,-0.015247367000000001,-0.023798302,0.018181108999999997,0.031784783999999996,-0.035913728,-0.010905801000000001,0.017325836000000004,0.041116777,0.03951890500000001,0.026243734000000008,-0.012494884,-0.003804806,0.017366462,0.025758145,0.037682780000000006,0.046627728,0.040425739,0.0047081240000000045,0.023060782,-0.050380155999999995,0.035289637,0.010805727000000003,0.039316747000000006,-0.053905957000000004,-0.009165108000000002,0.022203689999999998,-0.0010149120000000013,0.046805251000000006,0.004242532000000002,0.007719719999999999,0.0022003320000000036,0.010190485999999999,-0.05418347600000001,-0.040294599,-0.007390141000000001,-0.041925221,0.011897573000000003,-0.019697960000000004,-0.005456089999999999,-0.028636612999999998,-0.000675387,-0.03539743},
{0.007895197949999998,0.0389296716,0.00586257705,-0.023347697699999996,-0.0335661916,0.030160924600000007,0.0489274991,-0.037521436699999994,0.0358725339,0.025576808800000007,-0.028682975320000005,-0.03306342599999999,0.04002010809999999,0.01746705063,-0.028783805199999995,-0.0316050603,-0.020855842899999998,-0.035000659499999996,-0.008161285959880001,-0.021444417897,0.009149058060000001,-0.04689517959999999,-0.026746590700000007,-0.0335649553,-0.03138411149999999,-0.030218051399999994,-0.021662585800000002,0.028882473399999997,-0.024623856,-0.0434879387,-0.0318111763,0.0576955584,0.03529669414,-0.030870281700000002,-0.0047851425599999995,-0.034986972500000005,0.04024152245,-0.01396879221,0.04806010248,-0.0366214238,0.0175644083,0.019793257400000007,-0.02904503713,0.035674983199999996,-0.023603044999999996,0.00780449652,0.0388899699,0.027248602999999993,-0.009814947485999998,0.037478875499999995},
{0.007851706,-0.027411268,-0.017869753999999998,0.04063672100000001,0.022251452,-0.034008096,-0.03262017799999999,0.039027145,-0.005787088000000003,-0.037980745,-0.0010335770000000008,0.023981844000000006,-0.03619586200000001,-0.015892420999999997,0.016175778999999994,0.029118970999999993,0.04970316500000001,0.024177813,-0.005686112000000001,0.014775970000000003,0.0048020680000000005,0.027579781000000005,0.044262996,0.039912369,0.046673906,0.005150093999999999,0.022695753999999995,-0.049212083,0.038299043,0.007670716999999996,0.038727,-0.043659616,-0.010499815999999999,0.01852748799999999,0.00830963,0.04627128799999998,0.002649103,0.008160044,0.008930235,0.014025770000000003,-0.048882626000000005,-0.04876424400000001,-0.0048558709999999995,-0.053288607999999994,0.022421132,-0.017339072,-0.019817832000000004,-0.03772597700000001,-0.004140474999999999,-0.024933288999999997},
};
const float thresholds[num_labels] = {1.9993667310102896,1.9993718653739545,1.9985220684139038};
