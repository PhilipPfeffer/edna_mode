??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
?
AudioMicrofrontend	
audio
filterbanks"out_type"
sample_rateint?}"
window_sizeint"
window_stepint
"
num_channelsint " 
upper_band_limitfloat% `?E" 
lower_band_limitfloat%  ?B"
smoothing_bitsint
"
even_smoothingfloat%???<"
odd_smoothingfloat%??u="$
min_signal_remainingfloat%??L="
enable_pcanbool( "
pcan_strengthfloat%33s?"
pcan_offsetfloat%  ?B"
	gain_bitsint"

enable_logbool("
scale_shiftint"
left_contextint "
right_contextint "
frame_strideint"
zero_paddingbool( "
	out_scaleint"
out_typetype0:
2
p
AudioSpectrogram	
input
spectrogram"
window_sizeint"
strideint"
magnitude_squaredbool( 
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
	DecodeWav
contents	
audio
sample_rate"$
desired_channelsint?????????"#
desired_samplesint?????????
?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*1.15.02v1.15.0-rc3-22-g590d6ee??

I
wav_dataPlaceholder*
dtype0*
_output_shapes
: *
shape: 
}
decoded_sample_data	DecodeWavwav_data*!
_output_shapes
:	?}: *
desired_samples?}*
desired_channels
?
AudioSpectrogramAudioSpectrogramdecoded_sample_data*
window_size?*#
_output_shapes
:b?*
stride?*
magnitude_squared(
J
Mul/yConst*
_output_shapes
: *
valueB
 * ??F*
dtype0
P
MulMuldecoded_sample_dataMul/y*
_output_shapes
:	?}*
T0
Z
CastCastMul*

SrcT0*

DstT0*
_output_shapes
:	?}*
Truncate( 
`
Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
[
ReshapeReshapeCastReshape/shape*
Tshape0*
T0*
_output_shapes	
:?}
?
AudioMicrofrontendAudioMicrofrontendReshape*
window_size*
	gain_bits*
min_signal_remaining%??L=*
_output_shapes

:b(*
right_context *
upper_band_limit% `?E*
sample_rate?}*
out_type0*
even_smoothing%???<*
pcan_strength%33s?*
lower_band_limit%  ?B*
left_context *
zero_padding( *
frame_stride*
	out_scale*

enable_log(*
scale_shift*
window_step
*
odd_smoothing%??u=*
smoothing_bits
*
enable_pcan(*
num_channels(*
pcan_offset%  ?B
L
Mul_1/yConst*
_output_shapes
: *
valueB
 *   =*
dtype0
R
Mul_1MulAudioMicrofrontendMul_1/y*
T0*
_output_shapes

:b(
`
Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"????P  
d
	Reshape_1ReshapeMul_1Reshape_1/shape*
_output_shapes
:	?*
Tshape0*
T0
h
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????b   (      
o
	Reshape_2Reshape	Reshape_1Reshape_2/shape*&
_output_shapes
:b(*
T0*
Tshape0
?
=MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights
?
;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/minConst*
dtype0*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
valueB
 *HY??*
_output_shapes
: 
?
;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/maxConst*
valueB
 *HY?>*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
dtype0*
_output_shapes
: 
?
EMobilenetV1/Conv2d_0/weights/Initializer/random_uniform/RandomUniformRandomUniform=MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/shape*&
_output_shapes
:*
T0*
seed2 */
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*

seed *
dtype0
?
;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/subSub;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/max;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/min*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
_output_shapes
: *
T0
?
;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/mulMulEMobilenetV1/Conv2d_0/weights/Initializer/random_uniform/RandomUniform;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights
?
7MobilenetV1/Conv2d_0/weights/Initializer/random_uniformAdd;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/mul;MobilenetV1/Conv2d_0/weights/Initializer/random_uniform/min*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*&
_output_shapes
:*
T0
?
MobilenetV1/Conv2d_0/weights
VariableV2*
	container *&
_output_shapes
:*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
shape:*
dtype0*
shared_name 
?
#MobilenetV1/Conv2d_0/weights/AssignAssignMobilenetV1/Conv2d_0/weights7MobilenetV1/Conv2d_0/weights/Initializer/random_uniform*&
_output_shapes
:*
T0*
validate_shape(*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
use_locking(
?
!MobilenetV1/Conv2d_0/weights/readIdentityMobilenetV1/Conv2d_0/weights*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
T0*&
_output_shapes
:
?
-MobilenetV1/Conv2d_0/biases/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*.
_class$
" loc:@MobilenetV1/Conv2d_0/biases
?
MobilenetV1/Conv2d_0/biases
VariableV2*
	container *
shape:*
_output_shapes
:*
dtype0*
shared_name *.
_class$
" loc:@MobilenetV1/Conv2d_0/biases
?
"MobilenetV1/Conv2d_0/biases/AssignAssignMobilenetV1/Conv2d_0/biases-MobilenetV1/Conv2d_0/biases/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*.
_class$
" loc:@MobilenetV1/Conv2d_0/biases
?
 MobilenetV1/Conv2d_0/biases/readIdentityMobilenetV1/Conv2d_0/biases*.
_class$
" loc:@MobilenetV1/Conv2d_0/biases*
_output_shapes
:*
T0

.MobilenetV1/MobilenetV1/Conv2d_0/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
'MobilenetV1/MobilenetV1/Conv2d_0/Conv2DConv2D	Reshape_2!MobilenetV1/Conv2d_0/weights/read*
data_formatNHWC*
paddingSAME*&
_output_shapes
:1*
use_cudnn_on_gpu(*
explicit_paddings
 *
T0*
	dilations
*
strides

?
(MobilenetV1/MobilenetV1/Conv2d_0/BiasAddBiasAdd'MobilenetV1/MobilenetV1/Conv2d_0/Conv2D MobilenetV1/Conv2d_0/biases/read*&
_output_shapes
:1*
data_formatNHWC*
T0
?
%MobilenetV1/MobilenetV1/Conv2d_0/ReluRelu(MobilenetV1/MobilenetV1/Conv2d_0/BiasAdd*
T0*&
_output_shapes
:1
?
QMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*%
valueB"            *
_output_shapes
:
?
OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *HY??*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
valueB
 *HY?>*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
_output_shapes
: *
dtype0
?
YMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/shape*
dtype0*
seed2 *C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*&
_output_shapes
:*
T0*

seed 
?
OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
T0
?
KMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*&
_output_shapes
:*
T0
?
0MobilenetV1/Conv2d_1_depthwise/depthwise_weights
VariableV2*
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
shape:*
	container *&
_output_shapes
:*
shared_name 
?
7MobilenetV1/Conv2d_1_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsKMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform*
use_locking(*
T0*
validate_shape(*&
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights
?
5MobilenetV1/Conv2d_1_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_1_depthwise/depthwise_weights*&
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/ShapeConst*
dtype0*
_output_shapes
:*%
valueB"            
?
BMobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
4MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwiseDepthwiseConv2dNative%MobilenetV1/MobilenetV1/Conv2d_0/Relu5MobilenetV1/Conv2d_1_depthwise/depthwise_weights/read*
T0*&
_output_shapes
:1*
	dilations
*
strides
*
paddingSAME*
data_formatNHWC
?
7MobilenetV1/Conv2d_1_depthwise/biases/Initializer/zerosConst*
dtype0*
valueB*    *8
_class.
,*loc:@MobilenetV1/Conv2d_1_depthwise/biases*
_output_shapes
:
?
%MobilenetV1/Conv2d_1_depthwise/biases
VariableV2*
dtype0*
shape:*
shared_name *8
_class.
,*loc:@MobilenetV1/Conv2d_1_depthwise/biases*
	container *
_output_shapes
:
?
,MobilenetV1/Conv2d_1_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_1_depthwise/biases7MobilenetV1/Conv2d_1_depthwise/biases/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_1_depthwise/biases*
T0
?
*MobilenetV1/Conv2d_1_depthwise/biases/readIdentity%MobilenetV1/Conv2d_1_depthwise/biases*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_1_depthwise/biases*
_output_shapes
:
?
2MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise*MobilenetV1/Conv2d_1_depthwise/biases/read*
data_formatNHWC*
T0*&
_output_shapes
:1
?
/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_1_depthwise/BiasAdd*&
_output_shapes
:1*
T0
?
GMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            *9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights
?
EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/minConst*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
valueB
 *   ?*
dtype0*
_output_shapes
: 
?
EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
dtype0*
_output_shapes
: *
valueB
 *   ?
?
OMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/shape*
seed2 *
T0*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*&
_output_shapes
:*

seed 
?
EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights
?
EMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/sub*&
_output_shapes
:*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
T0
?
AMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
T0*&
_output_shapes
:
?
&MobilenetV1/Conv2d_1_pointwise/weights
VariableV2*
	container *9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
shape:*&
_output_shapes
:*
dtype0*
shared_name 
?
-MobilenetV1/Conv2d_1_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_1_pointwise/weightsAMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
T0*&
_output_shapes
:*
use_locking(*
validate_shape(
?
+MobilenetV1/Conv2d_1_pointwise/weights/readIdentity&MobilenetV1/Conv2d_1_pointwise/weights*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*&
_output_shapes
:
?
7MobilenetV1/Conv2d_1_pointwise/biases/Initializer/zerosConst*
valueB*    *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_1_pointwise/biases*
_output_shapes
:
?
%MobilenetV1/Conv2d_1_pointwise/biases
VariableV2*
	container *
dtype0*
_output_shapes
:*
shape:*
shared_name *8
_class.
,*loc:@MobilenetV1/Conv2d_1_pointwise/biases
?
,MobilenetV1/Conv2d_1_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_1_pointwise/biases7MobilenetV1/Conv2d_1_pointwise/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_1_pointwise/biases*
use_locking(*
_output_shapes
:*
T0
?
*MobilenetV1/Conv2d_1_pointwise/biases/readIdentity%MobilenetV1/Conv2d_1_pointwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_1_pointwise/biases*
T0*
_output_shapes
:
?
8MobilenetV1/MobilenetV1/Conv2d_1_pointwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
1MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu+MobilenetV1/Conv2d_1_pointwise/weights/read*
data_formatNHWC*
paddingSAME*
T0*
explicit_paddings
 *
use_cudnn_on_gpu(*&
_output_shapes
:1*
	dilations
*
strides

?
2MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D*MobilenetV1/Conv2d_1_pointwise/biases/read*
T0*&
_output_shapes
:1*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_1_pointwise/BiasAdd*&
_output_shapes
:1*
T0
?
QMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0*%
valueB"            *
_output_shapes
:
?
OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
valueB
 *??J?*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
dtype0
?
OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
valueB
 *??J>*
dtype0
?
YMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/shape*
dtype0*&
_output_shapes
:*

seed *
seed2 *
T0*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
_output_shapes
: *
T0
?
OMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights
?
KMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform/min*&
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
T0
?
0MobilenetV1/Conv2d_2_depthwise/depthwise_weights
VariableV2*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
shared_name *&
_output_shapes
:*
shape:*
	container *
dtype0
?
7MobilenetV1/Conv2d_2_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsKMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform*
use_locking(*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*&
_output_shapes
:*
T0*
validate_shape(
?
5MobilenetV1/Conv2d_2_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_2_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
T0*&
_output_shapes
:
?
:MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0
?
BMobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
4MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu5MobilenetV1/Conv2d_2_depthwise/depthwise_weights/read*
	dilations
*
paddingSAME*&
_output_shapes
:
*
strides
*
T0*
data_formatNHWC
?
7MobilenetV1/Conv2d_2_depthwise/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_depthwise/biases*
valueB*    *
_output_shapes
:
?
%MobilenetV1/Conv2d_2_depthwise/biases
VariableV2*
	container *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_depthwise/biases*
_output_shapes
:*
shape:*
shared_name 
?
,MobilenetV1/Conv2d_2_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_2_depthwise/biases7MobilenetV1/Conv2d_2_depthwise/biases/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_depthwise/biases
?
*MobilenetV1/Conv2d_2_depthwise/biases/readIdentity%MobilenetV1/Conv2d_2_depthwise/biases*
T0*
_output_shapes
:*8
_class.
,*loc:@MobilenetV1/Conv2d_2_depthwise/biases
?
2MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise*MobilenetV1/Conv2d_2_depthwise/biases/read*
T0*&
_output_shapes
:
*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_2_depthwise/BiasAdd*&
_output_shapes
:
*
T0
?
GMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*%
valueB"             *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights
?
EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *???*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights
?
EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
dtype0*
valueB
 *??>
?
OMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/shape*

seed *&
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
T0*
seed2 *
dtype0
?
EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
T0*
_output_shapes
: 
?
EMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/sub*
T0*&
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights
?
AMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform/min*
T0*&
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights
?
&MobilenetV1/Conv2d_2_pointwise/weights
VariableV2*
shared_name *9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*&
_output_shapes
: *
dtype0*
	container *
shape: 
?
-MobilenetV1/Conv2d_2_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_2_pointwise/weightsAMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*&
_output_shapes
: 
?
+MobilenetV1/Conv2d_2_pointwise/weights/readIdentity&MobilenetV1/Conv2d_2_pointwise/weights*&
_output_shapes
: *
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights
?
7MobilenetV1/Conv2d_2_pointwise/biases/Initializer/zerosConst*
_output_shapes
: *
valueB *    *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_pointwise/biases
?
%MobilenetV1/Conv2d_2_pointwise/biases
VariableV2*
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_pointwise/biases*
shape: *
	container *
shared_name *
_output_shapes
: 
?
,MobilenetV1/Conv2d_2_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_2_pointwise/biases7MobilenetV1/Conv2d_2_pointwise/biases/Initializer/zeros*
_output_shapes
: *8
_class.
,*loc:@MobilenetV1/Conv2d_2_pointwise/biases*
validate_shape(*
T0*
use_locking(
?
*MobilenetV1/Conv2d_2_pointwise/biases/readIdentity%MobilenetV1/Conv2d_2_pointwise/biases*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_pointwise/biases*
_output_shapes
: 
?
8MobilenetV1/MobilenetV1/Conv2d_2_pointwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
1MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu+MobilenetV1/Conv2d_2_pointwise/weights/read*
explicit_paddings
 *
strides
*
use_cudnn_on_gpu(*
data_formatNHWC*
T0*
	dilations
*&
_output_shapes
:
 *
paddingSAME
?
2MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D*MobilenetV1/Conv2d_2_pointwise/biases/read*
data_formatNHWC*&
_output_shapes
:
 *
T0
?
/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_2_pointwise/BiasAdd*
T0*&
_output_shapes
:
 
?
QMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"             *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
_output_shapes
:
?
OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
dtype0*
valueB
 *???
?
OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
valueB
 *??>
?
YMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/shape*
dtype0*
seed2 *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*&
_output_shapes
: *

seed *
T0
?
OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
_output_shapes
: *
T0
?
OMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*&
_output_shapes
: 
?
KMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform/min*&
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
T0
?
0MobilenetV1/Conv2d_3_depthwise/depthwise_weights
VariableV2*
dtype0*
shape: *
	container *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*&
_output_shapes
: *
shared_name 
?
7MobilenetV1/Conv2d_3_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsKMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform*
use_locking(*
validate_shape(*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
T0*&
_output_shapes
: 
?
5MobilenetV1/Conv2d_3_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_3_depthwise/depthwise_weights*&
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/ShapeConst*
_output_shapes
:*%
valueB"             *
dtype0
?
BMobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
4MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu5MobilenetV1/Conv2d_3_depthwise/depthwise_weights/read*
T0*
paddingSAME*
	dilations
*
strides
*
data_formatNHWC*&
_output_shapes
:
 
?
7MobilenetV1/Conv2d_3_depthwise/biases/Initializer/zerosConst*
_output_shapes
: *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_3_depthwise/biases*
valueB *    
?
%MobilenetV1/Conv2d_3_depthwise/biases
VariableV2*
shape: *
_output_shapes
: *
shared_name *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_3_depthwise/biases*
	container 
?
,MobilenetV1/Conv2d_3_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_3_depthwise/biases7MobilenetV1/Conv2d_3_depthwise/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_3_depthwise/biases*
use_locking(*
T0*
_output_shapes
: 
?
*MobilenetV1/Conv2d_3_depthwise/biases/readIdentity%MobilenetV1/Conv2d_3_depthwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_3_depthwise/biases*
T0*
_output_shapes
: 
?
2MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise*MobilenetV1/Conv2d_3_depthwise/biases/read*
data_formatNHWC*&
_output_shapes
:
 *
T0
?
/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_3_depthwise/BiasAdd*
T0*&
_output_shapes
:
 
?
GMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/shapeConst*%
valueB"              *
_output_shapes
:*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *qĜ?*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
dtype0*
_output_shapes
: *
valueB
 *qĜ>
?
OMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *

seed *
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*&
_output_shapes
:  
?
EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
EMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/sub*&
_output_shapes
:  *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
T0
?
AMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform/min*&
_output_shapes
:  *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
T0
?
&MobilenetV1/Conv2d_3_pointwise/weights
VariableV2*
dtype0*
shared_name *
	container *&
_output_shapes
:  *
shape:  *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
-MobilenetV1/Conv2d_3_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_3_pointwise/weightsAMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights*
validate_shape(*
T0*&
_output_shapes
:  
?
+MobilenetV1/Conv2d_3_pointwise/weights/readIdentity&MobilenetV1/Conv2d_3_pointwise/weights*&
_output_shapes
:  *
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
7MobilenetV1/Conv2d_3_pointwise/biases/Initializer/zerosConst*
valueB *    *8
_class.
,*loc:@MobilenetV1/Conv2d_3_pointwise/biases*
dtype0*
_output_shapes
: 
?
%MobilenetV1/Conv2d_3_pointwise/biases
VariableV2*
	container *
_output_shapes
: *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_3_pointwise/biases*
shared_name *
shape: 
?
,MobilenetV1/Conv2d_3_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_3_pointwise/biases7MobilenetV1/Conv2d_3_pointwise/biases/Initializer/zeros*
use_locking(*
_output_shapes
: *
T0*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_3_pointwise/biases
?
*MobilenetV1/Conv2d_3_pointwise/biases/readIdentity%MobilenetV1/Conv2d_3_pointwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_3_pointwise/biases*
_output_shapes
: *
T0
?
8MobilenetV1/MobilenetV1/Conv2d_3_pointwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
1MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu+MobilenetV1/Conv2d_3_pointwise/weights/read*
data_formatNHWC*
strides
*&
_output_shapes
:
 *
use_cudnn_on_gpu(*
explicit_paddings
 *
T0*
paddingSAME*
	dilations

?
2MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D*MobilenetV1/Conv2d_3_pointwise/biases/read*&
_output_shapes
:
 *
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_3_pointwise/BiasAdd*&
_output_shapes
:
 *
T0
?
QMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*%
valueB"             *
dtype0*
_output_shapes
:
?
OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
valueB
 *???*
dtype0*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
_output_shapes
: *
dtype0*
valueB
 *??>
?
YMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/shape*
seed2 *
T0*

seed *
dtype0*&
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*&
_output_shapes
: 
?
KMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*&
_output_shapes
: 
?
0MobilenetV1/Conv2d_4_depthwise/depthwise_weights
VariableV2*&
_output_shapes
: *
shared_name *
	container *
shape: *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
dtype0
?
7MobilenetV1/Conv2d_4_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsKMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*&
_output_shapes
: *
validate_shape(*
use_locking(
?
5MobilenetV1/Conv2d_4_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_4_depthwise/depthwise_weights*&
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             
?
BMobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
4MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu5MobilenetV1/Conv2d_4_depthwise/depthwise_weights/read*
data_formatNHWC*
paddingSAME*
strides
*
	dilations
*
T0*&
_output_shapes
: 
?
7MobilenetV1/Conv2d_4_depthwise/biases/Initializer/zerosConst*
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_4_depthwise/biases*
valueB *    *
_output_shapes
: 
?
%MobilenetV1/Conv2d_4_depthwise/biases
VariableV2*
dtype0*
	container *
shape: *
_output_shapes
: *8
_class.
,*loc:@MobilenetV1/Conv2d_4_depthwise/biases*
shared_name 
?
,MobilenetV1/Conv2d_4_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_4_depthwise/biases7MobilenetV1/Conv2d_4_depthwise/biases/Initializer/zeros*
use_locking(*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_4_depthwise/biases*
T0*
_output_shapes
: 
?
*MobilenetV1/Conv2d_4_depthwise/biases/readIdentity%MobilenetV1/Conv2d_4_depthwise/biases*
_output_shapes
: *
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_4_depthwise/biases
?
2MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise*MobilenetV1/Conv2d_4_depthwise/biases/read*&
_output_shapes
: *
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_4_depthwise/BiasAdd*&
_output_shapes
: *
T0
?
GMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*%
valueB"          @   *
dtype0*
_output_shapes
:
?
EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/minConst*
valueB
 *  ??*
dtype0*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights
?
EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
valueB
 *  ?>*
dtype0
?
OMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/shape*
seed2 *
T0*&
_output_shapes
: @*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
dtype0*

seed 
?
EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights
?
EMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/sub*&
_output_shapes
: @*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights
?
AMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform/min*&
_output_shapes
: @*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
T0
?
&MobilenetV1/Conv2d_4_pointwise/weights
VariableV2*
dtype0*
shape: @*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
	container *&
_output_shapes
: @*
shared_name 
?
-MobilenetV1/Conv2d_4_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_4_pointwise/weightsAMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*
use_locking(*
T0*&
_output_shapes
: @*
validate_shape(
?
+MobilenetV1/Conv2d_4_pointwise/weights/readIdentity&MobilenetV1/Conv2d_4_pointwise/weights*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*&
_output_shapes
: @
?
7MobilenetV1/Conv2d_4_pointwise/biases/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *8
_class.
,*loc:@MobilenetV1/Conv2d_4_pointwise/biases*
dtype0
?
%MobilenetV1/Conv2d_4_pointwise/biases
VariableV2*
	container *8
_class.
,*loc:@MobilenetV1/Conv2d_4_pointwise/biases*
shape:@*
_output_shapes
:@*
shared_name *
dtype0
?
,MobilenetV1/Conv2d_4_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_4_pointwise/biases7MobilenetV1/Conv2d_4_pointwise/biases/Initializer/zeros*
validate_shape(*
T0*
_output_shapes
:@*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_4_pointwise/biases
?
*MobilenetV1/Conv2d_4_pointwise/biases/readIdentity%MobilenetV1/Conv2d_4_pointwise/biases*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_4_pointwise/biases*
_output_shapes
:@
?
8MobilenetV1/MobilenetV1/Conv2d_4_pointwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
1MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu+MobilenetV1/Conv2d_4_pointwise/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*&
_output_shapes
:@*
	dilations
*
explicit_paddings
 *
paddingSAME*
data_formatNHWC
?
2MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D*MobilenetV1/Conv2d_4_pointwise/biases/read*&
_output_shapes
:@*
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_4_pointwise/BiasAdd*
T0*&
_output_shapes
:@
?
QMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*%
valueB"      @      *
_output_shapes
:*
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
dtype0*
valueB
 *?hϽ
?
OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?h?=*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
YMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/shape*

seed *
dtype0*
seed2 *C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
T0*&
_output_shapes
:@
?
OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
KMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*&
_output_shapes
:@
?
0MobilenetV1/Conv2d_5_depthwise/depthwise_weights
VariableV2*
shared_name *&
_output_shapes
:@*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
	container *
dtype0*
shape:@
?
7MobilenetV1/Conv2d_5_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsKMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform*
use_locking(*
validate_shape(*&
_output_shapes
:@*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
T0
?
5MobilenetV1/Conv2d_5_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_5_depthwise/depthwise_weights*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights*&
_output_shapes
:@
?
:MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      
?
BMobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
4MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu5MobilenetV1/Conv2d_5_depthwise/depthwise_weights/read*
T0*
strides
*
	dilations
*
paddingSAME*
data_formatNHWC*&
_output_shapes
:@
?
7MobilenetV1/Conv2d_5_depthwise/biases/Initializer/zerosConst*8
_class.
,*loc:@MobilenetV1/Conv2d_5_depthwise/biases*
dtype0*
_output_shapes
:@*
valueB@*    
?
%MobilenetV1/Conv2d_5_depthwise/biases
VariableV2*
shared_name *
	container *
_output_shapes
:@*
shape:@*8
_class.
,*loc:@MobilenetV1/Conv2d_5_depthwise/biases*
dtype0
?
,MobilenetV1/Conv2d_5_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_5_depthwise/biases7MobilenetV1/Conv2d_5_depthwise/biases/Initializer/zeros*
_output_shapes
:@*
use_locking(*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_5_depthwise/biases*
T0
?
*MobilenetV1/Conv2d_5_depthwise/biases/readIdentity%MobilenetV1/Conv2d_5_depthwise/biases*
_output_shapes
:@*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_5_depthwise/biases
?
2MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise*MobilenetV1/Conv2d_5_depthwise/biases/read*
data_formatNHWC*&
_output_shapes
:@*
T0
?
/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_5_depthwise/BiasAdd*
T0*&
_output_shapes
:@
?
GMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
_output_shapes
:*%
valueB"      @   @   *
dtype0
?
EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
valueB
 *׳]?
?
EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
valueB
 *׳]>*
dtype0*
_output_shapes
: 
?
OMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/shape*

seed *
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
seed2 *&
_output_shapes
:@@*
T0
?
EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
T0*
_output_shapes
: 
?
EMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/sub*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*&
_output_shapes
:@@
?
AMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*&
_output_shapes
:@@*
T0
?
&MobilenetV1/Conv2d_5_pointwise/weights
VariableV2*
shape:@@*
shared_name *
	container *9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*&
_output_shapes
:@@*
dtype0
?
-MobilenetV1/Conv2d_5_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_5_pointwise/weightsAMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*&
_output_shapes
:@@*
T0*
validate_shape(*
use_locking(
?
+MobilenetV1/Conv2d_5_pointwise/weights/readIdentity&MobilenetV1/Conv2d_5_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
T0*&
_output_shapes
:@@
?
7MobilenetV1/Conv2d_5_pointwise/biases/Initializer/zerosConst*8
_class.
,*loc:@MobilenetV1/Conv2d_5_pointwise/biases*
valueB@*    *
_output_shapes
:@*
dtype0
?
%MobilenetV1/Conv2d_5_pointwise/biases
VariableV2*
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_5_pointwise/biases*
_output_shapes
:@*
	container *
shared_name *
shape:@
?
,MobilenetV1/Conv2d_5_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_5_pointwise/biases7MobilenetV1/Conv2d_5_pointwise/biases/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@*8
_class.
,*loc:@MobilenetV1/Conv2d_5_pointwise/biases
?
*MobilenetV1/Conv2d_5_pointwise/biases/readIdentity%MobilenetV1/Conv2d_5_pointwise/biases*
T0*
_output_shapes
:@*8
_class.
,*loc:@MobilenetV1/Conv2d_5_pointwise/biases
?
8MobilenetV1/MobilenetV1/Conv2d_5_pointwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
1MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu+MobilenetV1/Conv2d_5_pointwise/weights/read*&
_output_shapes
:@*
strides
*
T0*
	dilations
*
paddingSAME*
explicit_paddings
 *
use_cudnn_on_gpu(*
data_formatNHWC
?
2MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D*MobilenetV1/Conv2d_5_pointwise/biases/read*&
_output_shapes
:@*
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_5_pointwise/BiasAdd*&
_output_shapes
:@*
T0
?
QMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*%
valueB"      @      *
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
_output_shapes
:
?
OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*
valueB
 *?hϽ
?
OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
valueB
 *?h?=*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*
_output_shapes
: 
?
YMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/shape*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
dtype0*&
_output_shapes
:@*

seed *
seed2 
?
OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/min*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
T0
?
OMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/sub*&
_output_shapes
:@*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
T0
?
KMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
T0*&
_output_shapes
:@
?
0MobilenetV1/Conv2d_6_depthwise/depthwise_weights
VariableV2*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
shared_name *
dtype0*
	container *
shape:@*&
_output_shapes
:@
?
7MobilenetV1/Conv2d_6_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsKMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform*&
_output_shapes
:@*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
use_locking(*
T0*
validate_shape(
?
5MobilenetV1/Conv2d_6_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_6_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*&
_output_shapes
:@*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/ShapeConst*
dtype0*%
valueB"      @      *
_output_shapes
:
?
BMobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
4MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu5MobilenetV1/Conv2d_6_depthwise/depthwise_weights/read*&
_output_shapes
:@*
T0*
paddingSAME*
strides
*
	dilations
*
data_formatNHWC
?
7MobilenetV1/Conv2d_6_depthwise/biases/Initializer/zerosConst*
dtype0*
valueB@*    *
_output_shapes
:@*8
_class.
,*loc:@MobilenetV1/Conv2d_6_depthwise/biases
?
%MobilenetV1/Conv2d_6_depthwise/biases
VariableV2*
	container *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_6_depthwise/biases*
shared_name *
shape:@*
_output_shapes
:@
?
,MobilenetV1/Conv2d_6_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_6_depthwise/biases7MobilenetV1/Conv2d_6_depthwise/biases/Initializer/zeros*8
_class.
,*loc:@MobilenetV1/Conv2d_6_depthwise/biases*
use_locking(*
T0*
_output_shapes
:@*
validate_shape(
?
*MobilenetV1/Conv2d_6_depthwise/biases/readIdentity%MobilenetV1/Conv2d_6_depthwise/biases*
_output_shapes
:@*8
_class.
,*loc:@MobilenetV1/Conv2d_6_depthwise/biases*
T0
?
2MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise*MobilenetV1/Conv2d_6_depthwise/biases/read*&
_output_shapes
:@*
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_6_depthwise/BiasAdd*&
_output_shapes
:@*
T0
?
GMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
dtype0*%
valueB"      @   ?   *
_output_shapes
:
?
EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
valueB
 *?5?
?
EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/maxConst*
valueB
 *?5>*
dtype0*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights
?
OMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/shape*'
_output_shapes
:@?*
dtype0*
seed2 *9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
T0*

seed 
?
EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
_output_shapes
: *
T0
?
EMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*'
_output_shapes
:@?*
T0
?
AMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
T0*'
_output_shapes
:@?
?
&MobilenetV1/Conv2d_6_pointwise/weights
VariableV2*
shared_name *
shape:@?*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*'
_output_shapes
:@?*
	container 
?
-MobilenetV1/Conv2d_6_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_6_pointwise/weightsAMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform*'
_output_shapes
:@?*
T0*
use_locking(*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights
?
+MobilenetV1/Conv2d_6_pointwise/weights/readIdentity&MobilenetV1/Conv2d_6_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
T0*'
_output_shapes
:@?
?
7MobilenetV1/Conv2d_6_pointwise/biases/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *8
_class.
,*loc:@MobilenetV1/Conv2d_6_pointwise/biases*
dtype0
?
%MobilenetV1/Conv2d_6_pointwise/biases
VariableV2*
shape:?*
shared_name *
_output_shapes	
:?*
	container *8
_class.
,*loc:@MobilenetV1/Conv2d_6_pointwise/biases*
dtype0
?
,MobilenetV1/Conv2d_6_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_6_pointwise/biases7MobilenetV1/Conv2d_6_pointwise/biases/Initializer/zeros*
T0*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_6_pointwise/biases*
validate_shape(*
_output_shapes	
:?
?
*MobilenetV1/Conv2d_6_pointwise/biases/readIdentity%MobilenetV1/Conv2d_6_pointwise/biases*
T0*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_6_pointwise/biases
?
8MobilenetV1/MobilenetV1/Conv2d_6_pointwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
1MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu+MobilenetV1/Conv2d_6_pointwise/weights/read*
strides
*
T0*
explicit_paddings
 *
use_cudnn_on_gpu(*
	dilations
*
data_formatNHWC*
paddingSAME*'
_output_shapes
:?
?
2MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D*MobilenetV1/Conv2d_6_pointwise/biases/read*
data_formatNHWC*'
_output_shapes
:?*
T0
?
/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_6_pointwise/BiasAdd*
T0*'
_output_shapes
:?
?
QMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      *C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
valueB
 *I:??*
_output_shapes
: *
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights
?
OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *I:?=*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights
?
YMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/shape*
T0*

seed *
dtype0*'
_output_shapes
:?*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
seed2 
?
OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
T0*
_output_shapes
: 
?
OMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*'
_output_shapes
:?
?
KMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*'
_output_shapes
:?
?
0MobilenetV1/Conv2d_7_depthwise/depthwise_weights
VariableV2*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
	container *'
_output_shapes
:?*
shared_name *
dtype0*
shape:?
?
7MobilenetV1/Conv2d_7_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsKMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
use_locking(*
validate_shape(*'
_output_shapes
:?
?
5MobilenetV1/Conv2d_7_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_7_depthwise/depthwise_weights*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*'
_output_shapes
:?*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/ShapeConst*
_output_shapes
:*%
valueB"      ?      *
dtype0
?
BMobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
4MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu5MobilenetV1/Conv2d_7_depthwise/depthwise_weights/read*
data_formatNHWC*'
_output_shapes
:?*
strides
*
paddingSAME*
T0*
	dilations

?
7MobilenetV1/Conv2d_7_depthwise/biases/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *8
_class.
,*loc:@MobilenetV1/Conv2d_7_depthwise/biases*
dtype0
?
%MobilenetV1/Conv2d_7_depthwise/biases
VariableV2*
shared_name *
_output_shapes	
:?*
dtype0*
shape:?*
	container *8
_class.
,*loc:@MobilenetV1/Conv2d_7_depthwise/biases
?
,MobilenetV1/Conv2d_7_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_7_depthwise/biases7MobilenetV1/Conv2d_7_depthwise/biases/Initializer/zeros*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_7_depthwise/biases*
T0*
_output_shapes	
:?*
validate_shape(
?
*MobilenetV1/Conv2d_7_depthwise/biases/readIdentity%MobilenetV1/Conv2d_7_depthwise/biases*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_7_depthwise/biases*
T0
?
2MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise*MobilenetV1/Conv2d_7_depthwise/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_7_depthwise/BiasAdd*'
_output_shapes
:?*
T0
?
GMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   *9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
?
EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/minConst*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
valueB
 *q??*
_output_shapes
: *
dtype0
?
EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
_output_shapes
: *
valueB
 *q?>
?
OMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/shape*

seed *9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
seed2 *(
_output_shapes
:??*
T0*
dtype0
?
EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
_output_shapes
: 
?
EMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
?
AMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
?
&MobilenetV1/Conv2d_7_pointwise/weights
VariableV2*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
	container *(
_output_shapes
:??*
shape:??*
shared_name 
?
-MobilenetV1/Conv2d_7_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_7_pointwise/weightsAMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform*
validate_shape(*
T0*
use_locking(*(
_output_shapes
:??*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
?
+MobilenetV1/Conv2d_7_pointwise/weights/readIdentity&MobilenetV1/Conv2d_7_pointwise/weights*(
_output_shapes
:??*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
?
7MobilenetV1/Conv2d_7_pointwise/biases/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_7_pointwise/biases
?
%MobilenetV1/Conv2d_7_pointwise/biases
VariableV2*
dtype0*
shape:?*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_7_pointwise/biases*
	container *
shared_name 
?
,MobilenetV1/Conv2d_7_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_7_pointwise/biases7MobilenetV1/Conv2d_7_pointwise/biases/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_7_pointwise/biases
?
*MobilenetV1/Conv2d_7_pointwise/biases/readIdentity%MobilenetV1/Conv2d_7_pointwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_7_pointwise/biases*
T0*
_output_shapes	
:?
?
8MobilenetV1/MobilenetV1/Conv2d_7_pointwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
1MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu+MobilenetV1/Conv2d_7_pointwise/weights/read*'
_output_shapes
:?*
T0*
use_cudnn_on_gpu(*
explicit_paddings
 *
data_formatNHWC*
strides
*
	dilations
*
paddingSAME
?
2MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D*MobilenetV1/Conv2d_7_pointwise/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_7_pointwise/BiasAdd*'
_output_shapes
:?*
T0
?
QMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*%
valueB"      ?      
?
OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
_output_shapes
: *
valueB
 *I:??*
dtype0
?
OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
valueB
 *I:?=*
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
dtype0
?
YMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/shape*

seed *'
_output_shapes
:?*
seed2 *
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
T0
?
OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
T0*
_output_shapes
: 
?
OMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*'
_output_shapes
:?
?
KMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform/min*'
_output_shapes
:?*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
T0
?
0MobilenetV1/Conv2d_8_depthwise/depthwise_weights
VariableV2*'
_output_shapes
:?*
	container *C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
shared_name *
dtype0*
shape:?
?
7MobilenetV1/Conv2d_8_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsKMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform*
validate_shape(*
use_locking(*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
T0*'
_output_shapes
:?
?
5MobilenetV1/Conv2d_8_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_8_depthwise/depthwise_weights*'
_output_shapes
:?*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/ShapeConst*
dtype0*%
valueB"      ?      *
_output_shapes
:
?
BMobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
4MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu5MobilenetV1/Conv2d_8_depthwise/depthwise_weights/read*
data_formatNHWC*
paddingSAME*
strides
*
	dilations
*
T0*'
_output_shapes
:?
?
7MobilenetV1/Conv2d_8_depthwise/biases/Initializer/zerosConst*8
_class.
,*loc:@MobilenetV1/Conv2d_8_depthwise/biases*
_output_shapes	
:?*
valueB?*    *
dtype0
?
%MobilenetV1/Conv2d_8_depthwise/biases
VariableV2*
shared_name *
shape:?*
	container *
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_8_depthwise/biases*
dtype0
?
,MobilenetV1/Conv2d_8_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_8_depthwise/biases7MobilenetV1/Conv2d_8_depthwise/biases/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_8_depthwise/biases
?
*MobilenetV1/Conv2d_8_depthwise/biases/readIdentity%MobilenetV1/Conv2d_8_depthwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_8_depthwise/biases*
T0*
_output_shapes	
:?
?
2MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise*MobilenetV1/Conv2d_8_depthwise/biases/read*'
_output_shapes
:?*
data_formatNHWC*
T0
?
/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_8_depthwise/BiasAdd*
T0*'
_output_shapes
:?
?
GMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*
_output_shapes
:*%
valueB"      ?   ?   
?
EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/minConst*
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
_output_shapes
: *
valueB
 *q??
?
EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *q?>*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights
?
OMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/shape*
dtype0*
seed2 *(
_output_shapes
:??*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*

seed 
?
EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/min*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
_output_shapes
: 
?
EMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/sub*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
T0*(
_output_shapes
:??
?
AMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform/min*(
_output_shapes
:??*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights
?
&MobilenetV1/Conv2d_8_pointwise/weights
VariableV2*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
dtype0*
shape:??*
	container *(
_output_shapes
:??*
shared_name 
?
-MobilenetV1/Conv2d_8_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_8_pointwise/weightsAMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform*
use_locking(*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
T0*(
_output_shapes
:??
?
+MobilenetV1/Conv2d_8_pointwise/weights/readIdentity&MobilenetV1/Conv2d_8_pointwise/weights*
T0*(
_output_shapes
:??*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights
?
7MobilenetV1/Conv2d_8_pointwise/biases/Initializer/zerosConst*8
_class.
,*loc:@MobilenetV1/Conv2d_8_pointwise/biases*
dtype0*
_output_shapes	
:?*
valueB?*    
?
%MobilenetV1/Conv2d_8_pointwise/biases
VariableV2*8
_class.
,*loc:@MobilenetV1/Conv2d_8_pointwise/biases*
	container *
_output_shapes	
:?*
dtype0*
shared_name *
shape:?
?
,MobilenetV1/Conv2d_8_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_8_pointwise/biases7MobilenetV1/Conv2d_8_pointwise/biases/Initializer/zeros*8
_class.
,*loc:@MobilenetV1/Conv2d_8_pointwise/biases*
_output_shapes	
:?*
T0*
use_locking(*
validate_shape(
?
*MobilenetV1/Conv2d_8_pointwise/biases/readIdentity%MobilenetV1/Conv2d_8_pointwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_8_pointwise/biases*
T0*
_output_shapes	
:?
?
8MobilenetV1/MobilenetV1/Conv2d_8_pointwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
1MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu+MobilenetV1/Conv2d_8_pointwise/weights/read*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*'
_output_shapes
:?*
	dilations
*
paddingSAME*
strides
*
explicit_paddings
 
?
2MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D*MobilenetV1/Conv2d_8_pointwise/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_8_pointwise/BiasAdd*'
_output_shapes
:?*
T0
?
QMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*%
valueB"      ?      
?
OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/minConst*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
_output_shapes
: *
dtype0*
valueB
 *I:??
?
OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *I:?=*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
dtype0
?
YMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformQMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/shape*
seed2 *'
_output_shapes
:?*
dtype0*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*

seed 
?
OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/subSubOMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/maxOMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
_output_shapes
: 
?
OMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/mulMulYMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformOMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/sub*'
_output_shapes
:?*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights
?
KMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniformAddOMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/mulOMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform/min*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*'
_output_shapes
:?*
T0
?
0MobilenetV1/Conv2d_9_depthwise/depthwise_weights
VariableV2*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
shape:?*
shared_name *'
_output_shapes
:?*
dtype0*
	container 
?
7MobilenetV1/Conv2d_9_depthwise/depthwise_weights/AssignAssign0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsKMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
use_locking(*'
_output_shapes
:?*
validate_shape(*
T0
?
5MobilenetV1/Conv2d_9_depthwise/depthwise_weights/readIdentity0MobilenetV1/Conv2d_9_depthwise/depthwise_weights*'
_output_shapes
:?*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
T0
?
:MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/ShapeConst*%
valueB"      ?      *
_output_shapes
:*
dtype0
?
BMobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
4MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu5MobilenetV1/Conv2d_9_depthwise/depthwise_weights/read*
paddingSAME*'
_output_shapes
:?*
data_formatNHWC*
T0*
strides
*
	dilations

?
7MobilenetV1/Conv2d_9_depthwise/biases/Initializer/zerosConst*
valueB?*    *
dtype0*8
_class.
,*loc:@MobilenetV1/Conv2d_9_depthwise/biases*
_output_shapes	
:?
?
%MobilenetV1/Conv2d_9_depthwise/biases
VariableV2*8
_class.
,*loc:@MobilenetV1/Conv2d_9_depthwise/biases*
	container *
dtype0*
_output_shapes	
:?*
shared_name *
shape:?
?
,MobilenetV1/Conv2d_9_depthwise/biases/AssignAssign%MobilenetV1/Conv2d_9_depthwise/biases7MobilenetV1/Conv2d_9_depthwise/biases/Initializer/zeros*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_9_depthwise/biases*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
*MobilenetV1/Conv2d_9_depthwise/biases/readIdentity%MobilenetV1/Conv2d_9_depthwise/biases*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_9_depthwise/biases*
_output_shapes	
:?
?
2MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BiasAddBiasAdd4MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise*MobilenetV1/Conv2d_9_depthwise/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_9_depthwise/BiasAdd*'
_output_shapes
:?*
T0
?
GMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*
_output_shapes
:
?
EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/minConst*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*
valueB
 *q??*
_output_shapes
: 
?
EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/maxConst*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
valueB
 *q?>*
dtype0*
_output_shapes
: 
?
OMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/shape*(
_output_shapes
:??*
T0*
seed2 *
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*

seed 
?
EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/subSubEMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/maxEMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/min*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
_output_shapes
: *
T0
?
EMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/mulMulOMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights
?
AMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniformAddEMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/mulEMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights
?
&MobilenetV1/Conv2d_9_pointwise/weights
VariableV2*
shape:??*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
dtype0*
shared_name *
	container *(
_output_shapes
:??
?
-MobilenetV1/Conv2d_9_pointwise/weights/AssignAssign&MobilenetV1/Conv2d_9_pointwise/weightsAMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
use_locking(*(
_output_shapes
:??*
validate_shape(*
T0
?
+MobilenetV1/Conv2d_9_pointwise/weights/readIdentity&MobilenetV1/Conv2d_9_pointwise/weights*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*(
_output_shapes
:??*
T0
?
7MobilenetV1/Conv2d_9_pointwise/biases/Initializer/zerosConst*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_9_pointwise/biases*
valueB?*    *
dtype0
?
%MobilenetV1/Conv2d_9_pointwise/biases
VariableV2*
_output_shapes	
:?*
	container *
shape:?*
shared_name *8
_class.
,*loc:@MobilenetV1/Conv2d_9_pointwise/biases*
dtype0
?
,MobilenetV1/Conv2d_9_pointwise/biases/AssignAssign%MobilenetV1/Conv2d_9_pointwise/biases7MobilenetV1/Conv2d_9_pointwise/biases/Initializer/zeros*8
_class.
,*loc:@MobilenetV1/Conv2d_9_pointwise/biases*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0
?
*MobilenetV1/Conv2d_9_pointwise/biases/readIdentity%MobilenetV1/Conv2d_9_pointwise/biases*8
_class.
,*loc:@MobilenetV1/Conv2d_9_pointwise/biases*
T0*
_output_shapes	
:?
?
8MobilenetV1/MobilenetV1/Conv2d_9_pointwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
1MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2DConv2D/MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu+MobilenetV1/Conv2d_9_pointwise/weights/read*
T0*
	dilations
*
strides
*
explicit_paddings
 *'
_output_shapes
:?*
paddingSAME*
use_cudnn_on_gpu(*
data_formatNHWC
?
2MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BiasAddBiasAdd1MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D*MobilenetV1/Conv2d_9_pointwise/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/ReluRelu2MobilenetV1/MobilenetV1/Conv2d_9_pointwise/BiasAdd*
T0*'
_output_shapes
:?
?
RMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
_output_shapes
:*
dtype0*%
valueB"      ?      
?
PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
dtype0*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
valueB
 *I:??*
_output_shapes
: 
?
PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *I:?=*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights
?
ZMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformRMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/shape*'
_output_shapes
:?*
dtype0*
T0*
seed2 *

seed *D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights
?
PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/subSubPMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/maxPMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/min*
_output_shapes
: *D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
T0
?
PMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/mulMulZMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformPMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*'
_output_shapes
:?
?
LMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniformAddPMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/mulPMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform/min*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
T0
?
1MobilenetV1/Conv2d_10_depthwise/depthwise_weights
VariableV2*
shared_name *D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
	container *'
_output_shapes
:?*
dtype0*
shape:?
?
8MobilenetV1/Conv2d_10_depthwise/depthwise_weights/AssignAssign1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsLMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform*
validate_shape(*
use_locking(*
T0*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights
?
6MobilenetV1/Conv2d_10_depthwise/depthwise_weights/readIdentity1MobilenetV1/Conv2d_10_depthwise/depthwise_weights*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
T0*'
_output_shapes
:?
?
;MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/ShapeConst*
dtype0*%
valueB"      ?      *
_output_shapes
:
?
CMobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwiseDepthwiseConv2dNative/MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6MobilenetV1/Conv2d_10_depthwise/depthwise_weights/read*'
_output_shapes
:?*
T0*
strides
*
paddingSAME*
data_formatNHWC*
	dilations

?
8MobilenetV1/Conv2d_10_depthwise/biases/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_10_depthwise/biases
?
&MobilenetV1/Conv2d_10_depthwise/biases
VariableV2*
shape:?*
	container *
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@MobilenetV1/Conv2d_10_depthwise/biases*
dtype0
?
-MobilenetV1/Conv2d_10_depthwise/biases/AssignAssign&MobilenetV1/Conv2d_10_depthwise/biases8MobilenetV1/Conv2d_10_depthwise/biases/Initializer/zeros*
_output_shapes	
:?*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_10_depthwise/biases*
use_locking(*
validate_shape(
?
+MobilenetV1/Conv2d_10_depthwise/biases/readIdentity&MobilenetV1/Conv2d_10_depthwise/biases*9
_class/
-+loc:@MobilenetV1/Conv2d_10_depthwise/biases*
T0*
_output_shapes	
:?
?
3MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BiasAddBiasAdd5MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise+MobilenetV1/Conv2d_10_depthwise/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
0MobilenetV1/MobilenetV1/Conv2d_10_depthwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_10_depthwise/BiasAdd*'
_output_shapes
:?*
T0
?
HMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights
?
FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/minConst*
valueB
 *q??*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
_output_shapes
: *
dtype0
?
FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *q?>*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights
?
PMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformHMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/shape*

seed *(
_output_shapes
:??*
T0*
dtype0*
seed2 *:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights
?
FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/subSubFMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/maxFMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/min*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
T0*
_output_shapes
: 
?
FMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/mulMulPMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/RandomUniformFMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights
?
BMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniformAddFMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/mulFMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform/min*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
T0
?
'MobilenetV1/Conv2d_10_pointwise/weights
VariableV2*
dtype0*
shared_name *
	container *
shape:??*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*(
_output_shapes
:??
?
.MobilenetV1/Conv2d_10_pointwise/weights/AssignAssign'MobilenetV1/Conv2d_10_pointwise/weightsBMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform*
T0*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
use_locking(*
validate_shape(
?
,MobilenetV1/Conv2d_10_pointwise/weights/readIdentity'MobilenetV1/Conv2d_10_pointwise/weights*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*(
_output_shapes
:??
?
8MobilenetV1/Conv2d_10_pointwise/biases/Initializer/zerosConst*
valueB?*    *
dtype0*9
_class/
-+loc:@MobilenetV1/Conv2d_10_pointwise/biases*
_output_shapes	
:?
?
&MobilenetV1/Conv2d_10_pointwise/biases
VariableV2*
shape:?*
shared_name *
	container *
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_10_pointwise/biases*
dtype0
?
-MobilenetV1/Conv2d_10_pointwise/biases/AssignAssign&MobilenetV1/Conv2d_10_pointwise/biases8MobilenetV1/Conv2d_10_pointwise/biases/Initializer/zeros*9
_class/
-+loc:@MobilenetV1/Conv2d_10_pointwise/biases*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
+MobilenetV1/Conv2d_10_pointwise/biases/readIdentity&MobilenetV1/Conv2d_10_pointwise/biases*9
_class/
-+loc:@MobilenetV1/Conv2d_10_pointwise/biases*
T0*
_output_shapes	
:?
?
9MobilenetV1/MobilenetV1/Conv2d_10_pointwise/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
?
2MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2DConv2D0MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu,MobilenetV1/Conv2d_10_pointwise/weights/read*
strides
*'
_output_shapes
:?*
T0*
explicit_paddings
 *
data_formatNHWC*
paddingSAME*
	dilations
*
use_cudnn_on_gpu(
?
3MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BiasAddBiasAdd2MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D+MobilenetV1/Conv2d_10_pointwise/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
0MobilenetV1/MobilenetV1/Conv2d_10_pointwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_10_pointwise/BiasAdd*
T0*'
_output_shapes
:?
?
RMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
_output_shapes
:*
dtype0*%
valueB"      ?      
?
PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *I:??*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
_output_shapes
: 
?
PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
dtype0*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
valueB
 *I:?=*
_output_shapes
: 
?
ZMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformRMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/shape*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*'
_output_shapes
:?*
T0*
seed2 *

seed 
?
PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/subSubPMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/maxPMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/min*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
T0*
_output_shapes
: 
?
PMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/mulMulZMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformPMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/sub*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*'
_output_shapes
:?
?
LMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniformAddPMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/mulPMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform/min*'
_output_shapes
:?*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights
?
1MobilenetV1/Conv2d_11_depthwise/depthwise_weights
VariableV2*'
_output_shapes
:?*
	container *D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
dtype0*
shape:?*
shared_name 
?
8MobilenetV1/Conv2d_11_depthwise/depthwise_weights/AssignAssign1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsLMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
validate_shape(*'
_output_shapes
:?*
use_locking(*
T0
?
6MobilenetV1/Conv2d_11_depthwise/depthwise_weights/readIdentity1MobilenetV1/Conv2d_11_depthwise/depthwise_weights*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
T0*'
_output_shapes
:?
?
;MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/ShapeConst*
dtype0*%
valueB"      ?      *
_output_shapes
:
?
CMobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
5MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwiseDepthwiseConv2dNative0MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6MobilenetV1/Conv2d_11_depthwise/depthwise_weights/read*
T0*
	dilations
*
strides
*
paddingSAME*'
_output_shapes
:?*
data_formatNHWC
?
8MobilenetV1/Conv2d_11_depthwise/biases/Initializer/zerosConst*9
_class/
-+loc:@MobilenetV1/Conv2d_11_depthwise/biases*
valueB?*    *
dtype0*
_output_shapes	
:?
?
&MobilenetV1/Conv2d_11_depthwise/biases
VariableV2*
_output_shapes	
:?*
shape:?*
dtype0*
	container *9
_class/
-+loc:@MobilenetV1/Conv2d_11_depthwise/biases*
shared_name 
?
-MobilenetV1/Conv2d_11_depthwise/biases/AssignAssign&MobilenetV1/Conv2d_11_depthwise/biases8MobilenetV1/Conv2d_11_depthwise/biases/Initializer/zeros*9
_class/
-+loc:@MobilenetV1/Conv2d_11_depthwise/biases*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
+MobilenetV1/Conv2d_11_depthwise/biases/readIdentity&MobilenetV1/Conv2d_11_depthwise/biases*
T0*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_11_depthwise/biases
?
3MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BiasAddBiasAdd5MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise+MobilenetV1/Conv2d_11_depthwise/biases/read*
data_formatNHWC*
T0*'
_output_shapes
:?
?
0MobilenetV1/MobilenetV1/Conv2d_11_depthwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_11_depthwise/BiasAdd*'
_output_shapes
:?*
T0
?
HMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"      ?   ?   *:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
_output_shapes
:
?
FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/minConst*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
dtype0*
valueB
 *q??*
_output_shapes
: 
?
FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/maxConst*
dtype0*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
valueB
 *q?>*
_output_shapes
: 
?
PMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformHMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/shape*

seed *
T0*(
_output_shapes
:??*
dtype0*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
seed2 
?
FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/subSubFMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/maxFMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights
?
FMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/mulMulPMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/RandomUniformFMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/sub*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*(
_output_shapes
:??*
T0
?
BMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniformAddFMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/mulFMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights
?
'MobilenetV1/Conv2d_11_pointwise/weights
VariableV2*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
shared_name *(
_output_shapes
:??*
dtype0*
	container *
shape:??
?
.MobilenetV1/Conv2d_11_pointwise/weights/AssignAssign'MobilenetV1/Conv2d_11_pointwise/weightsBMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform*
T0*
validate_shape(*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
use_locking(
?
,MobilenetV1/Conv2d_11_pointwise/weights/readIdentity'MobilenetV1/Conv2d_11_pointwise/weights*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*
T0*(
_output_shapes
:??
?
8MobilenetV1/Conv2d_11_pointwise/biases/Initializer/zerosConst*9
_class/
-+loc:@MobilenetV1/Conv2d_11_pointwise/biases*
dtype0*
valueB?*    *
_output_shapes	
:?
?
&MobilenetV1/Conv2d_11_pointwise/biases
VariableV2*9
_class/
-+loc:@MobilenetV1/Conv2d_11_pointwise/biases*
dtype0*
shared_name *
_output_shapes	
:?*
shape:?*
	container 
?
-MobilenetV1/Conv2d_11_pointwise/biases/AssignAssign&MobilenetV1/Conv2d_11_pointwise/biases8MobilenetV1/Conv2d_11_pointwise/biases/Initializer/zeros*9
_class/
-+loc:@MobilenetV1/Conv2d_11_pointwise/biases*
_output_shapes	
:?*
use_locking(*
validate_shape(*
T0
?
+MobilenetV1/Conv2d_11_pointwise/biases/readIdentity&MobilenetV1/Conv2d_11_pointwise/biases*
T0*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_11_pointwise/biases
?
9MobilenetV1/MobilenetV1/Conv2d_11_pointwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
2MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2DConv2D0MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu,MobilenetV1/Conv2d_11_pointwise/weights/read*'
_output_shapes
:?*
strides
*
explicit_paddings
 *
paddingSAME*
use_cudnn_on_gpu(*
data_formatNHWC*
	dilations
*
T0
?
3MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BiasAddBiasAdd2MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D+MobilenetV1/Conv2d_11_pointwise/biases/read*'
_output_shapes
:?*
T0*
data_formatNHWC
?
0MobilenetV1/MobilenetV1/Conv2d_11_pointwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_11_pointwise/BiasAdd*
T0*'
_output_shapes
:?
?
RMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*%
valueB"      ?      
?
PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *I:??*
_output_shapes
: *D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
valueB
 *I:?=*
dtype0*
_output_shapes
: *D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
ZMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformRMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/shape*
T0*
seed2 *
dtype0*'
_output_shapes
:?*

seed *D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/subSubPMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/maxPMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
PMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/mulMulZMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformPMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/sub*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
T0*'
_output_shapes
:?
?
LMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniformAddPMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/mulPMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform/min*
T0*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
1MobilenetV1/Conv2d_12_depthwise/depthwise_weights
VariableV2*
shape:?*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*'
_output_shapes
:?*
	container *
shared_name *
dtype0
?
8MobilenetV1/Conv2d_12_depthwise/depthwise_weights/AssignAssign1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsLMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform*
validate_shape(*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
use_locking(*
T0
?
6MobilenetV1/Conv2d_12_depthwise/depthwise_weights/readIdentity1MobilenetV1/Conv2d_12_depthwise/depthwise_weights*'
_output_shapes
:?*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights
?
;MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/ShapeConst*
_output_shapes
:*%
valueB"      ?      *
dtype0
?
CMobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
?
5MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwiseDepthwiseConv2dNative0MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6MobilenetV1/Conv2d_12_depthwise/depthwise_weights/read*
	dilations
*
T0*
paddingSAME*
strides
*
data_formatNHWC*'
_output_shapes
:?
?
8MobilenetV1/Conv2d_12_depthwise/biases/Initializer/zerosConst*
valueB?*    *9
_class/
-+loc:@MobilenetV1/Conv2d_12_depthwise/biases*
dtype0*
_output_shapes	
:?
?
&MobilenetV1/Conv2d_12_depthwise/biases
VariableV2*
dtype0*
	container *
shared_name *
shape:?*9
_class/
-+loc:@MobilenetV1/Conv2d_12_depthwise/biases*
_output_shapes	
:?
?
-MobilenetV1/Conv2d_12_depthwise/biases/AssignAssign&MobilenetV1/Conv2d_12_depthwise/biases8MobilenetV1/Conv2d_12_depthwise/biases/Initializer/zeros*9
_class/
-+loc:@MobilenetV1/Conv2d_12_depthwise/biases*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
+MobilenetV1/Conv2d_12_depthwise/biases/readIdentity&MobilenetV1/Conv2d_12_depthwise/biases*9
_class/
-+loc:@MobilenetV1/Conv2d_12_depthwise/biases*
_output_shapes	
:?*
T0
?
3MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BiasAddBiasAdd5MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise+MobilenetV1/Conv2d_12_depthwise/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
0MobilenetV1/MobilenetV1/Conv2d_12_depthwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_12_depthwise/BiasAdd*
T0*'
_output_shapes
:?
?
HMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/shapeConst*%
valueB"      ?      *
_output_shapes
:*
dtype0*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights
?
FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/minConst*
dtype0*
valueB
 *   ?*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
_output_shapes
: 
?
FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *   >*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
dtype0
?
PMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformHMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/shape*

seed *
dtype0*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
T0*
seed2 
?
FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/subSubFMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/maxFMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/min*
T0*
_output_shapes
: *:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights
?
FMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/mulMulPMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/RandomUniformFMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
T0
?
BMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniformAddFMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/mulFMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform/min*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
T0*(
_output_shapes
:??
?
'MobilenetV1/Conv2d_12_pointwise/weights
VariableV2*
dtype0*
shared_name *:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*(
_output_shapes
:??*
shape:??*
	container 
?
.MobilenetV1/Conv2d_12_pointwise/weights/AssignAssign'MobilenetV1/Conv2d_12_pointwise/weightsBMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
use_locking(*
validate_shape(*(
_output_shapes
:??
?
,MobilenetV1/Conv2d_12_pointwise/weights/readIdentity'MobilenetV1/Conv2d_12_pointwise/weights*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*(
_output_shapes
:??*
T0
?
8MobilenetV1/Conv2d_12_pointwise/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_12_pointwise/biases*
valueB?*    
?
&MobilenetV1/Conv2d_12_pointwise/biases
VariableV2*
	container *
dtype0*
_output_shapes	
:?*
shared_name *9
_class/
-+loc:@MobilenetV1/Conv2d_12_pointwise/biases*
shape:?
?
-MobilenetV1/Conv2d_12_pointwise/biases/AssignAssign&MobilenetV1/Conv2d_12_pointwise/biases8MobilenetV1/Conv2d_12_pointwise/biases/Initializer/zeros*
T0*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_12_pointwise/biases*
_output_shapes	
:?*
validate_shape(
?
+MobilenetV1/Conv2d_12_pointwise/biases/readIdentity&MobilenetV1/Conv2d_12_pointwise/biases*
_output_shapes	
:?*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_12_pointwise/biases
?
9MobilenetV1/MobilenetV1/Conv2d_12_pointwise/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
?
2MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2DConv2D0MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu,MobilenetV1/Conv2d_12_pointwise/weights/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
strides
*'
_output_shapes
:?*
T0*
data_formatNHWC*
	dilations

?
3MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BiasAddBiasAdd2MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D+MobilenetV1/Conv2d_12_pointwise/biases/read*
data_formatNHWC*'
_output_shapes
:?*
T0
?
0MobilenetV1/MobilenetV1/Conv2d_12_pointwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_12_pointwise/BiasAdd*
T0*'
_output_shapes
:?
?
RMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/shapeConst*
dtype0*%
valueB"            *D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
_output_shapes
:
?
PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/minConst*
valueB
 *??P?*
dtype0*
_output_shapes
: *D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *??P=*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
_output_shapes
: 
?
ZMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformRandomUniformRMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/shape*'
_output_shapes
:?*
dtype0*

seed *
T0*
seed2 *D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/subSubPMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/maxPMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/min*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
T0*
_output_shapes
: 
?
PMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/mulMulZMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/RandomUniformPMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/sub*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
T0
?
LMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniformAddPMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/mulPMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform/min*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
T0*'
_output_shapes
:?
?
1MobilenetV1/Conv2d_13_depthwise/depthwise_weights
VariableV2*
shape:?*
dtype0*'
_output_shapes
:?*
shared_name *
	container *D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
8MobilenetV1/Conv2d_13_depthwise/depthwise_weights/AssignAssign1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsLMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
T0*
use_locking(*'
_output_shapes
:?*
validate_shape(
?
6MobilenetV1/Conv2d_13_depthwise/depthwise_weights/readIdentity1MobilenetV1/Conv2d_13_depthwise/depthwise_weights*
T0*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
;MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:
?
CMobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwiseDepthwiseConv2dNative0MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6MobilenetV1/Conv2d_13_depthwise/depthwise_weights/read*
data_formatNHWC*'
_output_shapes
:?*
strides
*
T0*
paddingSAME*
	dilations

?
8MobilenetV1/Conv2d_13_depthwise/biases/Initializer/zerosConst*
valueB?*    *9
_class/
-+loc:@MobilenetV1/Conv2d_13_depthwise/biases*
dtype0*
_output_shapes	
:?
?
&MobilenetV1/Conv2d_13_depthwise/biases
VariableV2*9
_class/
-+loc:@MobilenetV1/Conv2d_13_depthwise/biases*
	container *
shared_name *
dtype0*
shape:?*
_output_shapes	
:?
?
-MobilenetV1/Conv2d_13_depthwise/biases/AssignAssign&MobilenetV1/Conv2d_13_depthwise/biases8MobilenetV1/Conv2d_13_depthwise/biases/Initializer/zeros*
T0*
use_locking(*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_13_depthwise/biases*
validate_shape(
?
+MobilenetV1/Conv2d_13_depthwise/biases/readIdentity&MobilenetV1/Conv2d_13_depthwise/biases*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_13_depthwise/biases*
_output_shapes	
:?
?
3MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BiasAddBiasAdd5MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise+MobilenetV1/Conv2d_13_depthwise/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:?
?
0MobilenetV1/MobilenetV1/Conv2d_13_depthwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_13_depthwise/BiasAdd*'
_output_shapes
:?*
T0
?
HMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights
?
FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/minConst*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
valueB
 *׳ݽ*
_output_shapes
: *
dtype0
?
FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *׳?=*
dtype0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights
?
PMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/RandomUniformRandomUniformHMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/shape*
T0*(
_output_shapes
:??*
dtype0*
seed2 *:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*

seed 
?
FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/subSubFMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/maxFMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/min*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
_output_shapes
: *
T0
?
FMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/mulMulPMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/RandomUniformFMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/sub*(
_output_shapes
:??*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights
?
BMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniformAddFMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/mulFMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform/min*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*(
_output_shapes
:??*
T0
?
'MobilenetV1/Conv2d_13_pointwise/weights
VariableV2*
dtype0*(
_output_shapes
:??*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
shape:??*
shared_name *
	container 
?
.MobilenetV1/Conv2d_13_pointwise/weights/AssignAssign'MobilenetV1/Conv2d_13_pointwise/weightsBMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform*(
_output_shapes
:??*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
validate_shape(*
use_locking(
?
,MobilenetV1/Conv2d_13_pointwise/weights/readIdentity'MobilenetV1/Conv2d_13_pointwise/weights*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*
T0*(
_output_shapes
:??
?
8MobilenetV1/Conv2d_13_pointwise/biases/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*
valueB?*    *9
_class/
-+loc:@MobilenetV1/Conv2d_13_pointwise/biases
?
&MobilenetV1/Conv2d_13_pointwise/biases
VariableV2*
	container *
_output_shapes	
:?*
shared_name *
dtype0*
shape:?*9
_class/
-+loc:@MobilenetV1/Conv2d_13_pointwise/biases
?
-MobilenetV1/Conv2d_13_pointwise/biases/AssignAssign&MobilenetV1/Conv2d_13_pointwise/biases8MobilenetV1/Conv2d_13_pointwise/biases/Initializer/zeros*
use_locking(*
T0*
_output_shapes	
:?*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_13_pointwise/biases
?
+MobilenetV1/Conv2d_13_pointwise/biases/readIdentity&MobilenetV1/Conv2d_13_pointwise/biases*9
_class/
-+loc:@MobilenetV1/Conv2d_13_pointwise/biases*
T0*
_output_shapes	
:?
?
9MobilenetV1/MobilenetV1/Conv2d_13_pointwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
?
2MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2DConv2D0MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu,MobilenetV1/Conv2d_13_pointwise/weights/read*'
_output_shapes
:?*
	dilations
*
use_cudnn_on_gpu(*
strides
*
explicit_paddings
 *
data_formatNHWC*
T0*
paddingSAME
?
3MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BiasAddBiasAdd2MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D+MobilenetV1/Conv2d_13_pointwise/biases/read*
T0*'
_output_shapes
:?*
data_formatNHWC
?
0MobilenetV1/MobilenetV1/Conv2d_13_pointwise/ReluRelu3MobilenetV1/MobilenetV1/Conv2d_13_pointwise/BiasAdd*'
_output_shapes
:?*
T0
?
#MobilenetV1/Embs/AvgPool_1a/AvgPoolAvgPool0MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu*
paddingVALID*
strides
*
T0*
ksize
*
data_formatNHWC*'
_output_shapes
:?
?
$MobilenetV1/Embs/Dropout_1b/IdentityIdentity#MobilenetV1/Embs/AvgPool_1a/AvgPool*
T0*'
_output_shapes
:?
?
GMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/shapeConst*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*%
valueB"         2   *
_output_shapes
:*
dtype0
?
EMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?c?*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
dtype0
?
EMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/maxConst*
dtype0*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
valueB
 *?c>*
_output_shapes
: 
?
OMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/RandomUniformRandomUniformGMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/shape*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
dtype0*
seed2 *

seed *'
_output_shapes
:?2*
T0
?
EMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/subSubEMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/maxEMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/min*
_output_shapes
: *9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
T0
?
EMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/mulMulOMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/RandomUniformEMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/sub*
T0*'
_output_shapes
:?2*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights
?
AMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniformAddEMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/mulEMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform/min*'
_output_shapes
:?2*
T0*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights
?
&MobilenetV1/Embs/Conv2d_1c_1x1/weights
VariableV2*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*'
_output_shapes
:?2*
shared_name *
	container *
shape:?2*
dtype0
?
-MobilenetV1/Embs/Conv2d_1c_1x1/weights/AssignAssign&MobilenetV1/Embs/Conv2d_1c_1x1/weightsAMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform*'
_output_shapes
:?2*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
T0*
use_locking(
?
+MobilenetV1/Embs/Conv2d_1c_1x1/weights/readIdentity&MobilenetV1/Embs/Conv2d_1c_1x1/weights*'
_output_shapes
:?2*
T0*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights
?
7MobilenetV1/Embs/Conv2d_1c_1x1/biases/Initializer/zerosConst*
valueB2*    *8
_class.
,*loc:@MobilenetV1/Embs/Conv2d_1c_1x1/biases*
dtype0*
_output_shapes
:2
?
%MobilenetV1/Embs/Conv2d_1c_1x1/biases
VariableV2*
_output_shapes
:2*8
_class.
,*loc:@MobilenetV1/Embs/Conv2d_1c_1x1/biases*
shared_name *
shape:2*
	container *
dtype0
?
,MobilenetV1/Embs/Conv2d_1c_1x1/biases/AssignAssign%MobilenetV1/Embs/Conv2d_1c_1x1/biases7MobilenetV1/Embs/Conv2d_1c_1x1/biases/Initializer/zeros*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Embs/Conv2d_1c_1x1/biases*
use_locking(*
_output_shapes
:2*
T0
?
*MobilenetV1/Embs/Conv2d_1c_1x1/biases/readIdentity%MobilenetV1/Embs/Conv2d_1c_1x1/biases*
_output_shapes
:2*8
_class.
,*loc:@MobilenetV1/Embs/Conv2d_1c_1x1/biases*
T0
}
,MobilenetV1/Embs/Conv2d_1c_1x1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
%MobilenetV1/Embs/Conv2d_1c_1x1/Conv2DConv2D$MobilenetV1/Embs/Dropout_1b/Identity+MobilenetV1/Embs/Conv2d_1c_1x1/weights/read*
strides
*
	dilations
*&
_output_shapes
:2*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME*
T0
?
&MobilenetV1/Embs/Conv2d_1c_1x1/BiasAddBiasAdd%MobilenetV1/Embs/Conv2d_1c_1x1/Conv2D*MobilenetV1/Embs/Conv2d_1c_1x1/biases/read*&
_output_shapes
:2*
T0*
data_formatNHWC
`
Reshape_3/shapeConst*
valueB"????2   *
dtype0*
_output_shapes
:
?
	Reshape_3Reshape&MobilenetV1/Embs/Conv2d_1c_1x1/BiasAddReshape_3/shape*
T0*
Tshape0*
_output_shapes

:2
M
labels_softmaxSoftmax	Reshape_3*
_output_shapes

:2*
T0
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
_output_shapes
: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*?
value?B?8BMobilenetV1/Conv2d_0/biasesBMobilenetV1/Conv2d_0/weightsB&MobilenetV1/Conv2d_10_depthwise/biasesB1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_10_pointwise/biasesB'MobilenetV1/Conv2d_10_pointwise/weightsB&MobilenetV1/Conv2d_11_depthwise/biasesB1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_11_pointwise/biasesB'MobilenetV1/Conv2d_11_pointwise/weightsB&MobilenetV1/Conv2d_12_depthwise/biasesB1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_12_pointwise/biasesB'MobilenetV1/Conv2d_12_pointwise/weightsB&MobilenetV1/Conv2d_13_depthwise/biasesB1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_13_pointwise/biasesB'MobilenetV1/Conv2d_13_pointwise/weightsB%MobilenetV1/Conv2d_1_depthwise/biasesB0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_1_pointwise/biasesB&MobilenetV1/Conv2d_1_pointwise/weightsB%MobilenetV1/Conv2d_2_depthwise/biasesB0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_2_pointwise/biasesB&MobilenetV1/Conv2d_2_pointwise/weightsB%MobilenetV1/Conv2d_3_depthwise/biasesB0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_3_pointwise/biasesB&MobilenetV1/Conv2d_3_pointwise/weightsB%MobilenetV1/Conv2d_4_depthwise/biasesB0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_4_pointwise/biasesB&MobilenetV1/Conv2d_4_pointwise/weightsB%MobilenetV1/Conv2d_5_depthwise/biasesB0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_5_pointwise/biasesB&MobilenetV1/Conv2d_5_pointwise/weightsB%MobilenetV1/Conv2d_6_depthwise/biasesB0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_6_pointwise/biasesB&MobilenetV1/Conv2d_6_pointwise/weightsB%MobilenetV1/Conv2d_7_depthwise/biasesB0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_7_pointwise/biasesB&MobilenetV1/Conv2d_7_pointwise/weightsB%MobilenetV1/Conv2d_8_depthwise/biasesB0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_8_pointwise/biasesB&MobilenetV1/Conv2d_8_pointwise/weightsB%MobilenetV1/Conv2d_9_depthwise/biasesB0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_9_pointwise/biasesB&MobilenetV1/Conv2d_9_pointwise/weightsB%MobilenetV1/Embs/Conv2d_1c_1x1/biasesB&MobilenetV1/Embs/Conv2d_1c_1x1/weights*
_output_shapes
:8*
dtype0
?
save/SaveV2/shape_and_slicesConst*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesMobilenetV1/Conv2d_0/biasesMobilenetV1/Conv2d_0/weights&MobilenetV1/Conv2d_10_depthwise/biases1MobilenetV1/Conv2d_10_depthwise/depthwise_weights&MobilenetV1/Conv2d_10_pointwise/biases'MobilenetV1/Conv2d_10_pointwise/weights&MobilenetV1/Conv2d_11_depthwise/biases1MobilenetV1/Conv2d_11_depthwise/depthwise_weights&MobilenetV1/Conv2d_11_pointwise/biases'MobilenetV1/Conv2d_11_pointwise/weights&MobilenetV1/Conv2d_12_depthwise/biases1MobilenetV1/Conv2d_12_depthwise/depthwise_weights&MobilenetV1/Conv2d_12_pointwise/biases'MobilenetV1/Conv2d_12_pointwise/weights&MobilenetV1/Conv2d_13_depthwise/biases1MobilenetV1/Conv2d_13_depthwise/depthwise_weights&MobilenetV1/Conv2d_13_pointwise/biases'MobilenetV1/Conv2d_13_pointwise/weights%MobilenetV1/Conv2d_1_depthwise/biases0MobilenetV1/Conv2d_1_depthwise/depthwise_weights%MobilenetV1/Conv2d_1_pointwise/biases&MobilenetV1/Conv2d_1_pointwise/weights%MobilenetV1/Conv2d_2_depthwise/biases0MobilenetV1/Conv2d_2_depthwise/depthwise_weights%MobilenetV1/Conv2d_2_pointwise/biases&MobilenetV1/Conv2d_2_pointwise/weights%MobilenetV1/Conv2d_3_depthwise/biases0MobilenetV1/Conv2d_3_depthwise/depthwise_weights%MobilenetV1/Conv2d_3_pointwise/biases&MobilenetV1/Conv2d_3_pointwise/weights%MobilenetV1/Conv2d_4_depthwise/biases0MobilenetV1/Conv2d_4_depthwise/depthwise_weights%MobilenetV1/Conv2d_4_pointwise/biases&MobilenetV1/Conv2d_4_pointwise/weights%MobilenetV1/Conv2d_5_depthwise/biases0MobilenetV1/Conv2d_5_depthwise/depthwise_weights%MobilenetV1/Conv2d_5_pointwise/biases&MobilenetV1/Conv2d_5_pointwise/weights%MobilenetV1/Conv2d_6_depthwise/biases0MobilenetV1/Conv2d_6_depthwise/depthwise_weights%MobilenetV1/Conv2d_6_pointwise/biases&MobilenetV1/Conv2d_6_pointwise/weights%MobilenetV1/Conv2d_7_depthwise/biases0MobilenetV1/Conv2d_7_depthwise/depthwise_weights%MobilenetV1/Conv2d_7_pointwise/biases&MobilenetV1/Conv2d_7_pointwise/weights%MobilenetV1/Conv2d_8_depthwise/biases0MobilenetV1/Conv2d_8_depthwise/depthwise_weights%MobilenetV1/Conv2d_8_pointwise/biases&MobilenetV1/Conv2d_8_pointwise/weights%MobilenetV1/Conv2d_9_depthwise/biases0MobilenetV1/Conv2d_9_depthwise/depthwise_weights%MobilenetV1/Conv2d_9_pointwise/biases&MobilenetV1/Conv2d_9_pointwise/weights%MobilenetV1/Embs/Conv2d_1c_1x1/biases&MobilenetV1/Embs/Conv2d_1c_1x1/weights*F
dtypes<
:28
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8BMobilenetV1/Conv2d_0/biasesBMobilenetV1/Conv2d_0/weightsB&MobilenetV1/Conv2d_10_depthwise/biasesB1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_10_pointwise/biasesB'MobilenetV1/Conv2d_10_pointwise/weightsB&MobilenetV1/Conv2d_11_depthwise/biasesB1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_11_pointwise/biasesB'MobilenetV1/Conv2d_11_pointwise/weightsB&MobilenetV1/Conv2d_12_depthwise/biasesB1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_12_pointwise/biasesB'MobilenetV1/Conv2d_12_pointwise/weightsB&MobilenetV1/Conv2d_13_depthwise/biasesB1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_13_pointwise/biasesB'MobilenetV1/Conv2d_13_pointwise/weightsB%MobilenetV1/Conv2d_1_depthwise/biasesB0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_1_pointwise/biasesB&MobilenetV1/Conv2d_1_pointwise/weightsB%MobilenetV1/Conv2d_2_depthwise/biasesB0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_2_pointwise/biasesB&MobilenetV1/Conv2d_2_pointwise/weightsB%MobilenetV1/Conv2d_3_depthwise/biasesB0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_3_pointwise/biasesB&MobilenetV1/Conv2d_3_pointwise/weightsB%MobilenetV1/Conv2d_4_depthwise/biasesB0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_4_pointwise/biasesB&MobilenetV1/Conv2d_4_pointwise/weightsB%MobilenetV1/Conv2d_5_depthwise/biasesB0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_5_pointwise/biasesB&MobilenetV1/Conv2d_5_pointwise/weightsB%MobilenetV1/Conv2d_6_depthwise/biasesB0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_6_pointwise/biasesB&MobilenetV1/Conv2d_6_pointwise/weightsB%MobilenetV1/Conv2d_7_depthwise/biasesB0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_7_pointwise/biasesB&MobilenetV1/Conv2d_7_pointwise/weightsB%MobilenetV1/Conv2d_8_depthwise/biasesB0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_8_pointwise/biasesB&MobilenetV1/Conv2d_8_pointwise/weightsB%MobilenetV1/Conv2d_9_depthwise/biasesB0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_9_pointwise/biasesB&MobilenetV1/Conv2d_9_pointwise/weightsB%MobilenetV1/Embs/Conv2d_1c_1x1/biasesB&MobilenetV1/Embs/Conv2d_1c_1x1/weights
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*F
dtypes<
:28*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::
?
save/AssignAssignMobilenetV1/Conv2d_0/biasessave/RestoreV2*
_output_shapes
:*
validate_shape(*.
_class$
" loc:@MobilenetV1/Conv2d_0/biases*
T0*
use_locking(
?
save/Assign_1AssignMobilenetV1/Conv2d_0/weightssave/RestoreV2:1*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*
T0*&
_output_shapes
:*
validate_shape(*
use_locking(
?
save/Assign_2Assign&MobilenetV1/Conv2d_10_depthwise/biasessave/RestoreV2:2*
validate_shape(*
_output_shapes	
:?*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_10_depthwise/biases*
T0
?
save/Assign_3Assign1MobilenetV1/Conv2d_10_depthwise/depthwise_weightssave/RestoreV2:3*
use_locking(*'
_output_shapes
:?*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
T0*
validate_shape(
?
save/Assign_4Assign&MobilenetV1/Conv2d_10_pointwise/biasessave/RestoreV2:4*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_10_pointwise/biases*
validate_shape(
?
save/Assign_5Assign'MobilenetV1/Conv2d_10_pointwise/weightssave/RestoreV2:5*
validate_shape(*
use_locking(*(
_output_shapes
:??*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights
?
save/Assign_6Assign&MobilenetV1/Conv2d_11_depthwise/biasessave/RestoreV2:6*
use_locking(*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_11_depthwise/biases*
T0*
_output_shapes	
:?
?
save/Assign_7Assign1MobilenetV1/Conv2d_11_depthwise/depthwise_weightssave/RestoreV2:7*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
use_locking(*'
_output_shapes
:?*
validate_shape(*
T0
?
save/Assign_8Assign&MobilenetV1/Conv2d_11_pointwise/biasessave/RestoreV2:8*9
_class/
-+loc:@MobilenetV1/Conv2d_11_pointwise/biases*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save/Assign_9Assign'MobilenetV1/Conv2d_11_pointwise/weightssave/RestoreV2:9*
T0*
use_locking(*(
_output_shapes
:??*
validate_shape(*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights
?
save/Assign_10Assign&MobilenetV1/Conv2d_12_depthwise/biasessave/RestoreV2:10*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_12_depthwise/biases
?
save/Assign_11Assign1MobilenetV1/Conv2d_12_depthwise/depthwise_weightssave/RestoreV2:11*
T0*
validate_shape(*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*'
_output_shapes
:?*
use_locking(
?
save/Assign_12Assign&MobilenetV1/Conv2d_12_pointwise/biasessave/RestoreV2:12*
validate_shape(*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_12_pointwise/biases*
_output_shapes	
:?*
use_locking(
?
save/Assign_13Assign'MobilenetV1/Conv2d_12_pointwise/weightssave/RestoreV2:13*
use_locking(*(
_output_shapes
:??*
validate_shape(*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
T0
?
save/Assign_14Assign&MobilenetV1/Conv2d_13_depthwise/biasessave/RestoreV2:14*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_13_depthwise/biases
?
save/Assign_15Assign1MobilenetV1/Conv2d_13_depthwise/depthwise_weightssave/RestoreV2:15*'
_output_shapes
:?*
use_locking(*
validate_shape(*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
save/Assign_16Assign&MobilenetV1/Conv2d_13_pointwise/biasessave/RestoreV2:16*9
_class/
-+loc:@MobilenetV1/Conv2d_13_pointwise/biases*
use_locking(*
validate_shape(*
T0*
_output_shapes	
:?
?
save/Assign_17Assign'MobilenetV1/Conv2d_13_pointwise/weightssave/RestoreV2:17*
use_locking(*
validate_shape(*(
_output_shapes
:??*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights
?
save/Assign_18Assign%MobilenetV1/Conv2d_1_depthwise/biasessave/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_1_depthwise/biases*
validate_shape(
?
save/Assign_19Assign0MobilenetV1/Conv2d_1_depthwise/depthwise_weightssave/RestoreV2:19*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights*
validate_shape(*&
_output_shapes
:*
T0*
use_locking(
?
save/Assign_20Assign%MobilenetV1/Conv2d_1_pointwise/biasessave/RestoreV2:20*
validate_shape(*
T0*
_output_shapes
:*8
_class.
,*loc:@MobilenetV1/Conv2d_1_pointwise/biases*
use_locking(
?
save/Assign_21Assign&MobilenetV1/Conv2d_1_pointwise/weightssave/RestoreV2:21*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
T0*
validate_shape(*&
_output_shapes
:
?
save/Assign_22Assign%MobilenetV1/Conv2d_2_depthwise/biasessave/RestoreV2:22*
validate_shape(*
use_locking(*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_depthwise/biases*
_output_shapes
:
?
save/Assign_23Assign0MobilenetV1/Conv2d_2_depthwise/depthwise_weightssave/RestoreV2:23*
validate_shape(*
T0*&
_output_shapes
:*
use_locking(*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights
?
save/Assign_24Assign%MobilenetV1/Conv2d_2_pointwise/biasessave/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
: *8
_class.
,*loc:@MobilenetV1/Conv2d_2_pointwise/biases*
T0
?
save/Assign_25Assign&MobilenetV1/Conv2d_2_pointwise/weightssave/RestoreV2:25*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*&
_output_shapes
: *
use_locking(*
validate_shape(
?
save/Assign_26Assign%MobilenetV1/Conv2d_3_depthwise/biasessave/RestoreV2:26*
validate_shape(*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_3_depthwise/biases*
_output_shapes
: *
T0
?
save/Assign_27Assign0MobilenetV1/Conv2d_3_depthwise/depthwise_weightssave/RestoreV2:27*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*&
_output_shapes
: *
use_locking(*
T0*
validate_shape(
?
save/Assign_28Assign%MobilenetV1/Conv2d_3_pointwise/biasessave/RestoreV2:28*
_output_shapes
: *
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_3_pointwise/biases*
validate_shape(*
T0
?
save/Assign_29Assign&MobilenetV1/Conv2d_3_pointwise/weightssave/RestoreV2:29*
T0*
validate_shape(*
use_locking(*&
_output_shapes
:  *9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
save/Assign_30Assign%MobilenetV1/Conv2d_4_depthwise/biasessave/RestoreV2:30*
T0*
_output_shapes
: *8
_class.
,*loc:@MobilenetV1/Conv2d_4_depthwise/biases*
validate_shape(*
use_locking(
?
save/Assign_31Assign0MobilenetV1/Conv2d_4_depthwise/depthwise_weightssave/RestoreV2:31*&
_output_shapes
: *
validate_shape(*C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights*
T0*
use_locking(
?
save/Assign_32Assign%MobilenetV1/Conv2d_4_pointwise/biasessave/RestoreV2:32*
use_locking(*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_4_pointwise/biases*
validate_shape(*
_output_shapes
:@
?
save/Assign_33Assign&MobilenetV1/Conv2d_4_pointwise/weightssave/RestoreV2:33*
validate_shape(*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights*&
_output_shapes
: @*
use_locking(
?
save/Assign_34Assign%MobilenetV1/Conv2d_5_depthwise/biasessave/RestoreV2:34*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_5_depthwise/biases
?
save/Assign_35Assign0MobilenetV1/Conv2d_5_depthwise/depthwise_weightssave/RestoreV2:35*&
_output_shapes
:@*
use_locking(*
T0*
validate_shape(*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
save/Assign_36Assign%MobilenetV1/Conv2d_5_pointwise/biasessave/RestoreV2:36*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@*8
_class.
,*loc:@MobilenetV1/Conv2d_5_pointwise/biases
?
save/Assign_37Assign&MobilenetV1/Conv2d_5_pointwise/weightssave/RestoreV2:37*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
T0*
validate_shape(*&
_output_shapes
:@@*
use_locking(
?
save/Assign_38Assign%MobilenetV1/Conv2d_6_depthwise/biasessave/RestoreV2:38*8
_class.
,*loc:@MobilenetV1/Conv2d_6_depthwise/biases*
T0*
use_locking(*
validate_shape(*
_output_shapes
:@
?
save/Assign_39Assign0MobilenetV1/Conv2d_6_depthwise/depthwise_weightssave/RestoreV2:39*&
_output_shapes
:@*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
T0*
use_locking(*
validate_shape(
?
save/Assign_40Assign%MobilenetV1/Conv2d_6_pointwise/biasessave/RestoreV2:40*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_6_pointwise/biases*
T0*
_output_shapes	
:?*
use_locking(
?
save/Assign_41Assign&MobilenetV1/Conv2d_6_pointwise/weightssave/RestoreV2:41*
validate_shape(*
T0*'
_output_shapes
:@?*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights
?
save/Assign_42Assign%MobilenetV1/Conv2d_7_depthwise/biasessave/RestoreV2:42*8
_class.
,*loc:@MobilenetV1/Conv2d_7_depthwise/biases*
use_locking(*
validate_shape(*
_output_shapes	
:?*
T0
?
save/Assign_43Assign0MobilenetV1/Conv2d_7_depthwise/depthwise_weightssave/RestoreV2:43*'
_output_shapes
:?*
validate_shape(*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights*
use_locking(
?
save/Assign_44Assign%MobilenetV1/Conv2d_7_pointwise/biasessave/RestoreV2:44*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_7_pointwise/biases*
use_locking(*
T0*
validate_shape(
?
save/Assign_45Assign&MobilenetV1/Conv2d_7_pointwise/weightssave/RestoreV2:45*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights*
use_locking(*
validate_shape(*(
_output_shapes
:??*
T0
?
save/Assign_46Assign%MobilenetV1/Conv2d_8_depthwise/biasessave/RestoreV2:46*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_8_depthwise/biases
?
save/Assign_47Assign0MobilenetV1/Conv2d_8_depthwise/depthwise_weightssave/RestoreV2:47*
validate_shape(*'
_output_shapes
:?*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
T0*
use_locking(
?
save/Assign_48Assign%MobilenetV1/Conv2d_8_pointwise/biasessave/RestoreV2:48*
validate_shape(*
T0*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_8_pointwise/biases*
_output_shapes	
:?
?
save/Assign_49Assign&MobilenetV1/Conv2d_8_pointwise/weightssave/RestoreV2:49*
validate_shape(*(
_output_shapes
:??*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
T0*
use_locking(
?
save/Assign_50Assign%MobilenetV1/Conv2d_9_depthwise/biasessave/RestoreV2:50*
T0*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_9_depthwise/biases*
use_locking(*
validate_shape(
?
save/Assign_51Assign0MobilenetV1/Conv2d_9_depthwise/depthwise_weightssave/RestoreV2:51*
validate_shape(*
T0*
use_locking(*'
_output_shapes
:?*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights
?
save/Assign_52Assign%MobilenetV1/Conv2d_9_pointwise/biasessave/RestoreV2:52*
validate_shape(*
T0*
_output_shapes	
:?*8
_class.
,*loc:@MobilenetV1/Conv2d_9_pointwise/biases*
use_locking(
?
save/Assign_53Assign&MobilenetV1/Conv2d_9_pointwise/weightssave/RestoreV2:53*
T0*(
_output_shapes
:??*
validate_shape(*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights
?
save/Assign_54Assign%MobilenetV1/Embs/Conv2d_1c_1x1/biasessave/RestoreV2:54*8
_class.
,*loc:@MobilenetV1/Embs/Conv2d_1c_1x1/biases*
T0*
validate_shape(*
use_locking(*
_output_shapes
:2
?
save/Assign_55Assign&MobilenetV1/Embs/Conv2d_1c_1x1/weightssave/RestoreV2:55*'
_output_shapes
:?2*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
use_locking(*
validate_shape(*
T0
?
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
dtype0*
shape: 
?
save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_62893a9743574af9a2f4646e4c6df832/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*?
value?B?8BMobilenetV1/Conv2d_0/biasesBMobilenetV1/Conv2d_0/weightsB&MobilenetV1/Conv2d_10_depthwise/biasesB1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_10_pointwise/biasesB'MobilenetV1/Conv2d_10_pointwise/weightsB&MobilenetV1/Conv2d_11_depthwise/biasesB1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_11_pointwise/biasesB'MobilenetV1/Conv2d_11_pointwise/weightsB&MobilenetV1/Conv2d_12_depthwise/biasesB1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_12_pointwise/biasesB'MobilenetV1/Conv2d_12_pointwise/weightsB&MobilenetV1/Conv2d_13_depthwise/biasesB1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_13_pointwise/biasesB'MobilenetV1/Conv2d_13_pointwise/weightsB%MobilenetV1/Conv2d_1_depthwise/biasesB0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_1_pointwise/biasesB&MobilenetV1/Conv2d_1_pointwise/weightsB%MobilenetV1/Conv2d_2_depthwise/biasesB0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_2_pointwise/biasesB&MobilenetV1/Conv2d_2_pointwise/weightsB%MobilenetV1/Conv2d_3_depthwise/biasesB0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_3_pointwise/biasesB&MobilenetV1/Conv2d_3_pointwise/weightsB%MobilenetV1/Conv2d_4_depthwise/biasesB0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_4_pointwise/biasesB&MobilenetV1/Conv2d_4_pointwise/weightsB%MobilenetV1/Conv2d_5_depthwise/biasesB0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_5_pointwise/biasesB&MobilenetV1/Conv2d_5_pointwise/weightsB%MobilenetV1/Conv2d_6_depthwise/biasesB0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_6_pointwise/biasesB&MobilenetV1/Conv2d_6_pointwise/weightsB%MobilenetV1/Conv2d_7_depthwise/biasesB0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_7_pointwise/biasesB&MobilenetV1/Conv2d_7_pointwise/weightsB%MobilenetV1/Conv2d_8_depthwise/biasesB0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_8_pointwise/biasesB&MobilenetV1/Conv2d_8_pointwise/weightsB%MobilenetV1/Conv2d_9_depthwise/biasesB0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_9_pointwise/biasesB&MobilenetV1/Conv2d_9_pointwise/weightsB%MobilenetV1/Embs/Conv2d_1c_1x1/biasesB&MobilenetV1/Embs/Conv2d_1c_1x1/weights*
dtype0
?
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:8*
dtype0
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesMobilenetV1/Conv2d_0/biasesMobilenetV1/Conv2d_0/weights&MobilenetV1/Conv2d_10_depthwise/biases1MobilenetV1/Conv2d_10_depthwise/depthwise_weights&MobilenetV1/Conv2d_10_pointwise/biases'MobilenetV1/Conv2d_10_pointwise/weights&MobilenetV1/Conv2d_11_depthwise/biases1MobilenetV1/Conv2d_11_depthwise/depthwise_weights&MobilenetV1/Conv2d_11_pointwise/biases'MobilenetV1/Conv2d_11_pointwise/weights&MobilenetV1/Conv2d_12_depthwise/biases1MobilenetV1/Conv2d_12_depthwise/depthwise_weights&MobilenetV1/Conv2d_12_pointwise/biases'MobilenetV1/Conv2d_12_pointwise/weights&MobilenetV1/Conv2d_13_depthwise/biases1MobilenetV1/Conv2d_13_depthwise/depthwise_weights&MobilenetV1/Conv2d_13_pointwise/biases'MobilenetV1/Conv2d_13_pointwise/weights%MobilenetV1/Conv2d_1_depthwise/biases0MobilenetV1/Conv2d_1_depthwise/depthwise_weights%MobilenetV1/Conv2d_1_pointwise/biases&MobilenetV1/Conv2d_1_pointwise/weights%MobilenetV1/Conv2d_2_depthwise/biases0MobilenetV1/Conv2d_2_depthwise/depthwise_weights%MobilenetV1/Conv2d_2_pointwise/biases&MobilenetV1/Conv2d_2_pointwise/weights%MobilenetV1/Conv2d_3_depthwise/biases0MobilenetV1/Conv2d_3_depthwise/depthwise_weights%MobilenetV1/Conv2d_3_pointwise/biases&MobilenetV1/Conv2d_3_pointwise/weights%MobilenetV1/Conv2d_4_depthwise/biases0MobilenetV1/Conv2d_4_depthwise/depthwise_weights%MobilenetV1/Conv2d_4_pointwise/biases&MobilenetV1/Conv2d_4_pointwise/weights%MobilenetV1/Conv2d_5_depthwise/biases0MobilenetV1/Conv2d_5_depthwise/depthwise_weights%MobilenetV1/Conv2d_5_pointwise/biases&MobilenetV1/Conv2d_5_pointwise/weights%MobilenetV1/Conv2d_6_depthwise/biases0MobilenetV1/Conv2d_6_depthwise/depthwise_weights%MobilenetV1/Conv2d_6_pointwise/biases&MobilenetV1/Conv2d_6_pointwise/weights%MobilenetV1/Conv2d_7_depthwise/biases0MobilenetV1/Conv2d_7_depthwise/depthwise_weights%MobilenetV1/Conv2d_7_pointwise/biases&MobilenetV1/Conv2d_7_pointwise/weights%MobilenetV1/Conv2d_8_depthwise/biases0MobilenetV1/Conv2d_8_depthwise/depthwise_weights%MobilenetV1/Conv2d_8_pointwise/biases&MobilenetV1/Conv2d_8_pointwise/weights%MobilenetV1/Conv2d_9_depthwise/biases0MobilenetV1/Conv2d_9_depthwise/depthwise_weights%MobilenetV1/Conv2d_9_pointwise/biases&MobilenetV1/Conv2d_9_pointwise/weights%MobilenetV1/Embs/Conv2d_1c_1x1/biases&MobilenetV1/Embs/Conv2d_1c_1x1/weights"/device:CPU:0*F
dtypes<
:28
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename*
T0
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*

axis *
N*
_output_shapes
:*
T0
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
?
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8BMobilenetV1/Conv2d_0/biasesBMobilenetV1/Conv2d_0/weightsB&MobilenetV1/Conv2d_10_depthwise/biasesB1MobilenetV1/Conv2d_10_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_10_pointwise/biasesB'MobilenetV1/Conv2d_10_pointwise/weightsB&MobilenetV1/Conv2d_11_depthwise/biasesB1MobilenetV1/Conv2d_11_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_11_pointwise/biasesB'MobilenetV1/Conv2d_11_pointwise/weightsB&MobilenetV1/Conv2d_12_depthwise/biasesB1MobilenetV1/Conv2d_12_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_12_pointwise/biasesB'MobilenetV1/Conv2d_12_pointwise/weightsB&MobilenetV1/Conv2d_13_depthwise/biasesB1MobilenetV1/Conv2d_13_depthwise/depthwise_weightsB&MobilenetV1/Conv2d_13_pointwise/biasesB'MobilenetV1/Conv2d_13_pointwise/weightsB%MobilenetV1/Conv2d_1_depthwise/biasesB0MobilenetV1/Conv2d_1_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_1_pointwise/biasesB&MobilenetV1/Conv2d_1_pointwise/weightsB%MobilenetV1/Conv2d_2_depthwise/biasesB0MobilenetV1/Conv2d_2_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_2_pointwise/biasesB&MobilenetV1/Conv2d_2_pointwise/weightsB%MobilenetV1/Conv2d_3_depthwise/biasesB0MobilenetV1/Conv2d_3_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_3_pointwise/biasesB&MobilenetV1/Conv2d_3_pointwise/weightsB%MobilenetV1/Conv2d_4_depthwise/biasesB0MobilenetV1/Conv2d_4_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_4_pointwise/biasesB&MobilenetV1/Conv2d_4_pointwise/weightsB%MobilenetV1/Conv2d_5_depthwise/biasesB0MobilenetV1/Conv2d_5_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_5_pointwise/biasesB&MobilenetV1/Conv2d_5_pointwise/weightsB%MobilenetV1/Conv2d_6_depthwise/biasesB0MobilenetV1/Conv2d_6_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_6_pointwise/biasesB&MobilenetV1/Conv2d_6_pointwise/weightsB%MobilenetV1/Conv2d_7_depthwise/biasesB0MobilenetV1/Conv2d_7_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_7_pointwise/biasesB&MobilenetV1/Conv2d_7_pointwise/weightsB%MobilenetV1/Conv2d_8_depthwise/biasesB0MobilenetV1/Conv2d_8_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_8_pointwise/biasesB&MobilenetV1/Conv2d_8_pointwise/weightsB%MobilenetV1/Conv2d_9_depthwise/biasesB0MobilenetV1/Conv2d_9_depthwise/depthwise_weightsB%MobilenetV1/Conv2d_9_pointwise/biasesB&MobilenetV1/Conv2d_9_pointwise/weightsB%MobilenetV1/Embs/Conv2d_1c_1x1/biasesB&MobilenetV1/Embs/Conv2d_1c_1x1/weights
?
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:8
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28
?
save_1/AssignAssignMobilenetV1/Conv2d_0/biasessave_1/RestoreV2*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*.
_class$
" loc:@MobilenetV1/Conv2d_0/biases
?
save_1/Assign_1AssignMobilenetV1/Conv2d_0/weightssave_1/RestoreV2:1*
T0*
use_locking(*
validate_shape(*/
_class%
#!loc:@MobilenetV1/Conv2d_0/weights*&
_output_shapes
:
?
save_1/Assign_2Assign&MobilenetV1/Conv2d_10_depthwise/biasessave_1/RestoreV2:2*
use_locking(*
_output_shapes	
:?*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_10_depthwise/biases*
T0
?
save_1/Assign_3Assign1MobilenetV1/Conv2d_10_depthwise/depthwise_weightssave_1/RestoreV2:3*'
_output_shapes
:?*
validate_shape(*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_10_depthwise/depthwise_weights*
use_locking(
?
save_1/Assign_4Assign&MobilenetV1/Conv2d_10_pointwise/biasessave_1/RestoreV2:4*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_10_pointwise/biases*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_5Assign'MobilenetV1/Conv2d_10_pointwise/weightssave_1/RestoreV2:5*(
_output_shapes
:??*
validate_shape(*:
_class0
.,loc:@MobilenetV1/Conv2d_10_pointwise/weights*
T0*
use_locking(
?
save_1/Assign_6Assign&MobilenetV1/Conv2d_11_depthwise/biasessave_1/RestoreV2:6*
validate_shape(*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_11_depthwise/biases*
T0*
use_locking(
?
save_1/Assign_7Assign1MobilenetV1/Conv2d_11_depthwise/depthwise_weightssave_1/RestoreV2:7*
validate_shape(*D
_class:
86loc:@MobilenetV1/Conv2d_11_depthwise/depthwise_weights*
T0*
use_locking(*'
_output_shapes
:?
?
save_1/Assign_8Assign&MobilenetV1/Conv2d_11_pointwise/biasessave_1/RestoreV2:8*
use_locking(*
T0*
_output_shapes	
:?*9
_class/
-+loc:@MobilenetV1/Conv2d_11_pointwise/biases*
validate_shape(
?
save_1/Assign_9Assign'MobilenetV1/Conv2d_11_pointwise/weightssave_1/RestoreV2:9*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@MobilenetV1/Conv2d_11_pointwise/weights*(
_output_shapes
:??
?
save_1/Assign_10Assign&MobilenetV1/Conv2d_12_depthwise/biasessave_1/RestoreV2:10*
_output_shapes	
:?*
use_locking(*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_12_depthwise/biases*
validate_shape(
?
save_1/Assign_11Assign1MobilenetV1/Conv2d_12_depthwise/depthwise_weightssave_1/RestoreV2:11*'
_output_shapes
:?*
T0*D
_class:
86loc:@MobilenetV1/Conv2d_12_depthwise/depthwise_weights*
validate_shape(*
use_locking(
?
save_1/Assign_12Assign&MobilenetV1/Conv2d_12_pointwise/biasessave_1/RestoreV2:12*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_12_pointwise/biases
?
save_1/Assign_13Assign'MobilenetV1/Conv2d_12_pointwise/weightssave_1/RestoreV2:13*(
_output_shapes
:??*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_12_pointwise/weights*
use_locking(*
validate_shape(
?
save_1/Assign_14Assign&MobilenetV1/Conv2d_13_depthwise/biasessave_1/RestoreV2:14*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_13_depthwise/biases
?
save_1/Assign_15Assign1MobilenetV1/Conv2d_13_depthwise/depthwise_weightssave_1/RestoreV2:15*
use_locking(*
T0*'
_output_shapes
:?*
validate_shape(*D
_class:
86loc:@MobilenetV1/Conv2d_13_depthwise/depthwise_weights
?
save_1/Assign_16Assign&MobilenetV1/Conv2d_13_pointwise/biasessave_1/RestoreV2:16*
_output_shapes	
:?*
T0*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_13_pointwise/biases*
validate_shape(
?
save_1/Assign_17Assign'MobilenetV1/Conv2d_13_pointwise/weightssave_1/RestoreV2:17*
use_locking(*
validate_shape(*
T0*:
_class0
.,loc:@MobilenetV1/Conv2d_13_pointwise/weights*(
_output_shapes
:??
?
save_1/Assign_18Assign%MobilenetV1/Conv2d_1_depthwise/biasessave_1/RestoreV2:18*
use_locking(*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_1_depthwise/biases*
_output_shapes
:*
validate_shape(
?
save_1/Assign_19Assign0MobilenetV1/Conv2d_1_depthwise/depthwise_weightssave_1/RestoreV2:19*&
_output_shapes
:*
T0*
use_locking(*
validate_shape(*C
_class9
75loc:@MobilenetV1/Conv2d_1_depthwise/depthwise_weights
?
save_1/Assign_20Assign%MobilenetV1/Conv2d_1_pointwise/biasessave_1/RestoreV2:20*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_1_pointwise/biases*
validate_shape(*
_output_shapes
:*
T0
?
save_1/Assign_21Assign&MobilenetV1/Conv2d_1_pointwise/weightssave_1/RestoreV2:21*9
_class/
-+loc:@MobilenetV1/Conv2d_1_pointwise/weights*
validate_shape(*&
_output_shapes
:*
T0*
use_locking(
?
save_1/Assign_22Assign%MobilenetV1/Conv2d_2_depthwise/biasessave_1/RestoreV2:22*
use_locking(*
_output_shapes
:*8
_class.
,*loc:@MobilenetV1/Conv2d_2_depthwise/biases*
T0*
validate_shape(
?
save_1/Assign_23Assign0MobilenetV1/Conv2d_2_depthwise/depthwise_weightssave_1/RestoreV2:23*
validate_shape(*&
_output_shapes
:*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_2_depthwise/depthwise_weights*
use_locking(
?
save_1/Assign_24Assign%MobilenetV1/Conv2d_2_pointwise/biasessave_1/RestoreV2:24*
use_locking(*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_2_pointwise/biases*
validate_shape(*
_output_shapes
: 
?
save_1/Assign_25Assign&MobilenetV1/Conv2d_2_pointwise/weightssave_1/RestoreV2:25*9
_class/
-+loc:@MobilenetV1/Conv2d_2_pointwise/weights*
T0*
use_locking(*&
_output_shapes
: *
validate_shape(
?
save_1/Assign_26Assign%MobilenetV1/Conv2d_3_depthwise/biasessave_1/RestoreV2:26*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_3_depthwise/biases*
_output_shapes
: *
use_locking(*
T0
?
save_1/Assign_27Assign0MobilenetV1/Conv2d_3_depthwise/depthwise_weightssave_1/RestoreV2:27*
validate_shape(*C
_class9
75loc:@MobilenetV1/Conv2d_3_depthwise/depthwise_weights*
use_locking(*
T0*&
_output_shapes
: 
?
save_1/Assign_28Assign%MobilenetV1/Conv2d_3_pointwise/biasessave_1/RestoreV2:28*8
_class.
,*loc:@MobilenetV1/Conv2d_3_pointwise/biases*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
?
save_1/Assign_29Assign&MobilenetV1/Conv2d_3_pointwise/weightssave_1/RestoreV2:29*
validate_shape(*&
_output_shapes
:  *
use_locking(*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_3_pointwise/weights
?
save_1/Assign_30Assign%MobilenetV1/Conv2d_4_depthwise/biasessave_1/RestoreV2:30*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_4_depthwise/biases
?
save_1/Assign_31Assign0MobilenetV1/Conv2d_4_depthwise/depthwise_weightssave_1/RestoreV2:31*
T0*
use_locking(*
validate_shape(*&
_output_shapes
: *C
_class9
75loc:@MobilenetV1/Conv2d_4_depthwise/depthwise_weights
?
save_1/Assign_32Assign%MobilenetV1/Conv2d_4_pointwise/biasessave_1/RestoreV2:32*
_output_shapes
:@*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_4_pointwise/biases*
T0*
validate_shape(
?
save_1/Assign_33Assign&MobilenetV1/Conv2d_4_pointwise/weightssave_1/RestoreV2:33*
validate_shape(*
use_locking(*&
_output_shapes
: @*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_4_pointwise/weights
?
save_1/Assign_34Assign%MobilenetV1/Conv2d_5_depthwise/biasessave_1/RestoreV2:34*
_output_shapes
:@*8
_class.
,*loc:@MobilenetV1/Conv2d_5_depthwise/biases*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_35Assign0MobilenetV1/Conv2d_5_depthwise/depthwise_weightssave_1/RestoreV2:35*&
_output_shapes
:@*
validate_shape(*
use_locking(*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_5_depthwise/depthwise_weights
?
save_1/Assign_36Assign%MobilenetV1/Conv2d_5_pointwise/biasessave_1/RestoreV2:36*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_5_pointwise/biases*
validate_shape(*
T0*
_output_shapes
:@
?
save_1/Assign_37Assign&MobilenetV1/Conv2d_5_pointwise/weightssave_1/RestoreV2:37*
use_locking(*&
_output_shapes
:@@*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_5_pointwise/weights*
T0
?
save_1/Assign_38Assign%MobilenetV1/Conv2d_6_depthwise/biasessave_1/RestoreV2:38*
validate_shape(*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_6_depthwise/biases*
_output_shapes
:@*
use_locking(
?
save_1/Assign_39Assign0MobilenetV1/Conv2d_6_depthwise/depthwise_weightssave_1/RestoreV2:39*C
_class9
75loc:@MobilenetV1/Conv2d_6_depthwise/depthwise_weights*
use_locking(*
validate_shape(*
T0*&
_output_shapes
:@
?
save_1/Assign_40Assign%MobilenetV1/Conv2d_6_pointwise/biasessave_1/RestoreV2:40*8
_class.
,*loc:@MobilenetV1/Conv2d_6_pointwise/biases*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_1/Assign_41Assign&MobilenetV1/Conv2d_6_pointwise/weightssave_1/RestoreV2:41*9
_class/
-+loc:@MobilenetV1/Conv2d_6_pointwise/weights*
validate_shape(*
use_locking(*
T0*'
_output_shapes
:@?
?
save_1/Assign_42Assign%MobilenetV1/Conv2d_7_depthwise/biasessave_1/RestoreV2:42*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_7_depthwise/biases*
_output_shapes	
:?*
validate_shape(*
use_locking(
?
save_1/Assign_43Assign0MobilenetV1/Conv2d_7_depthwise/depthwise_weightssave_1/RestoreV2:43*
validate_shape(*'
_output_shapes
:?*
use_locking(*
T0*C
_class9
75loc:@MobilenetV1/Conv2d_7_depthwise/depthwise_weights
?
save_1/Assign_44Assign%MobilenetV1/Conv2d_7_pointwise/biasessave_1/RestoreV2:44*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_7_pointwise/biases
?
save_1/Assign_45Assign&MobilenetV1/Conv2d_7_pointwise/weightssave_1/RestoreV2:45*
use_locking(*(
_output_shapes
:??*
T0*
validate_shape(*9
_class/
-+loc:@MobilenetV1/Conv2d_7_pointwise/weights
?
save_1/Assign_46Assign%MobilenetV1/Conv2d_8_depthwise/biasessave_1/RestoreV2:46*8
_class.
,*loc:@MobilenetV1/Conv2d_8_depthwise/biases*
validate_shape(*
use_locking(*
T0*
_output_shapes	
:?
?
save_1/Assign_47Assign0MobilenetV1/Conv2d_8_depthwise/depthwise_weightssave_1/RestoreV2:47*C
_class9
75loc:@MobilenetV1/Conv2d_8_depthwise/depthwise_weights*
use_locking(*
validate_shape(*
T0*'
_output_shapes
:?
?
save_1/Assign_48Assign%MobilenetV1/Conv2d_8_pointwise/biasessave_1/RestoreV2:48*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Conv2d_8_pointwise/biases
?
save_1/Assign_49Assign&MobilenetV1/Conv2d_8_pointwise/weightssave_1/RestoreV2:49*
use_locking(*9
_class/
-+loc:@MobilenetV1/Conv2d_8_pointwise/weights*
validate_shape(*(
_output_shapes
:??*
T0
?
save_1/Assign_50Assign%MobilenetV1/Conv2d_9_depthwise/biasessave_1/RestoreV2:50*
T0*8
_class.
,*loc:@MobilenetV1/Conv2d_9_depthwise/biases*
validate_shape(*
_output_shapes	
:?*
use_locking(
?
save_1/Assign_51Assign0MobilenetV1/Conv2d_9_depthwise/depthwise_weightssave_1/RestoreV2:51*
T0*'
_output_shapes
:?*
use_locking(*C
_class9
75loc:@MobilenetV1/Conv2d_9_depthwise/depthwise_weights*
validate_shape(
?
save_1/Assign_52Assign%MobilenetV1/Conv2d_9_pointwise/biasessave_1/RestoreV2:52*
use_locking(*8
_class.
,*loc:@MobilenetV1/Conv2d_9_pointwise/biases*
T0*
_output_shapes	
:?*
validate_shape(
?
save_1/Assign_53Assign&MobilenetV1/Conv2d_9_pointwise/weightssave_1/RestoreV2:53*
validate_shape(*(
_output_shapes
:??*
T0*9
_class/
-+loc:@MobilenetV1/Conv2d_9_pointwise/weights*
use_locking(
?
save_1/Assign_54Assign%MobilenetV1/Embs/Conv2d_1c_1x1/biasessave_1/RestoreV2:54*
T0*
use_locking(*
_output_shapes
:2*
validate_shape(*8
_class.
,*loc:@MobilenetV1/Embs/Conv2d_1c_1x1/biases
?
save_1/Assign_55Assign&MobilenetV1/Embs/Conv2d_1c_1x1/weightssave_1/RestoreV2:55*9
_class/
-+loc:@MobilenetV1/Embs/Conv2d_1c_1x1/weights*
T0*'
_output_shapes
:?2*
use_locking(*
validate_shape(
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard"?B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"?]
	variables?\?\
?
MobilenetV1/Conv2d_0/weights:0#MobilenetV1/Conv2d_0/weights/Assign#MobilenetV1/Conv2d_0/weights/read:029MobilenetV1/Conv2d_0/weights/Initializer/random_uniform:08
?
MobilenetV1/Conv2d_0/biases:0"MobilenetV1/Conv2d_0/biases/Assign"MobilenetV1/Conv2d_0/biases/read:02/MobilenetV1/Conv2d_0/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_1_depthwise/depthwise_weights:07MobilenetV1/Conv2d_1_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_1_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_1_depthwise/biases:0,MobilenetV1/Conv2d_1_depthwise/biases/Assign,MobilenetV1/Conv2d_1_depthwise/biases/read:029MobilenetV1/Conv2d_1_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_1_pointwise/weights:0-MobilenetV1/Conv2d_1_pointwise/weights/Assign-MobilenetV1/Conv2d_1_pointwise/weights/read:02CMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_1_pointwise/biases:0,MobilenetV1/Conv2d_1_pointwise/biases/Assign,MobilenetV1/Conv2d_1_pointwise/biases/read:029MobilenetV1/Conv2d_1_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_2_depthwise/depthwise_weights:07MobilenetV1/Conv2d_2_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_2_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_2_depthwise/biases:0,MobilenetV1/Conv2d_2_depthwise/biases/Assign,MobilenetV1/Conv2d_2_depthwise/biases/read:029MobilenetV1/Conv2d_2_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_2_pointwise/weights:0-MobilenetV1/Conv2d_2_pointwise/weights/Assign-MobilenetV1/Conv2d_2_pointwise/weights/read:02CMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_2_pointwise/biases:0,MobilenetV1/Conv2d_2_pointwise/biases/Assign,MobilenetV1/Conv2d_2_pointwise/biases/read:029MobilenetV1/Conv2d_2_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_3_depthwise/depthwise_weights:07MobilenetV1/Conv2d_3_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_3_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_3_depthwise/biases:0,MobilenetV1/Conv2d_3_depthwise/biases/Assign,MobilenetV1/Conv2d_3_depthwise/biases/read:029MobilenetV1/Conv2d_3_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_3_pointwise/weights:0-MobilenetV1/Conv2d_3_pointwise/weights/Assign-MobilenetV1/Conv2d_3_pointwise/weights/read:02CMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_3_pointwise/biases:0,MobilenetV1/Conv2d_3_pointwise/biases/Assign,MobilenetV1/Conv2d_3_pointwise/biases/read:029MobilenetV1/Conv2d_3_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_4_depthwise/depthwise_weights:07MobilenetV1/Conv2d_4_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_4_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_4_depthwise/biases:0,MobilenetV1/Conv2d_4_depthwise/biases/Assign,MobilenetV1/Conv2d_4_depthwise/biases/read:029MobilenetV1/Conv2d_4_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_4_pointwise/weights:0-MobilenetV1/Conv2d_4_pointwise/weights/Assign-MobilenetV1/Conv2d_4_pointwise/weights/read:02CMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_4_pointwise/biases:0,MobilenetV1/Conv2d_4_pointwise/biases/Assign,MobilenetV1/Conv2d_4_pointwise/biases/read:029MobilenetV1/Conv2d_4_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_5_depthwise/depthwise_weights:07MobilenetV1/Conv2d_5_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_5_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_5_depthwise/biases:0,MobilenetV1/Conv2d_5_depthwise/biases/Assign,MobilenetV1/Conv2d_5_depthwise/biases/read:029MobilenetV1/Conv2d_5_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_5_pointwise/weights:0-MobilenetV1/Conv2d_5_pointwise/weights/Assign-MobilenetV1/Conv2d_5_pointwise/weights/read:02CMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_5_pointwise/biases:0,MobilenetV1/Conv2d_5_pointwise/biases/Assign,MobilenetV1/Conv2d_5_pointwise/biases/read:029MobilenetV1/Conv2d_5_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_6_depthwise/depthwise_weights:07MobilenetV1/Conv2d_6_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_6_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_6_depthwise/biases:0,MobilenetV1/Conv2d_6_depthwise/biases/Assign,MobilenetV1/Conv2d_6_depthwise/biases/read:029MobilenetV1/Conv2d_6_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_6_pointwise/weights:0-MobilenetV1/Conv2d_6_pointwise/weights/Assign-MobilenetV1/Conv2d_6_pointwise/weights/read:02CMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_6_pointwise/biases:0,MobilenetV1/Conv2d_6_pointwise/biases/Assign,MobilenetV1/Conv2d_6_pointwise/biases/read:029MobilenetV1/Conv2d_6_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_7_depthwise/depthwise_weights:07MobilenetV1/Conv2d_7_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_7_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_7_depthwise/biases:0,MobilenetV1/Conv2d_7_depthwise/biases/Assign,MobilenetV1/Conv2d_7_depthwise/biases/read:029MobilenetV1/Conv2d_7_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_7_pointwise/weights:0-MobilenetV1/Conv2d_7_pointwise/weights/Assign-MobilenetV1/Conv2d_7_pointwise/weights/read:02CMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_7_pointwise/biases:0,MobilenetV1/Conv2d_7_pointwise/biases/Assign,MobilenetV1/Conv2d_7_pointwise/biases/read:029MobilenetV1/Conv2d_7_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_8_depthwise/depthwise_weights:07MobilenetV1/Conv2d_8_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_8_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_8_depthwise/biases:0,MobilenetV1/Conv2d_8_depthwise/biases/Assign,MobilenetV1/Conv2d_8_depthwise/biases/read:029MobilenetV1/Conv2d_8_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_8_pointwise/weights:0-MobilenetV1/Conv2d_8_pointwise/weights/Assign-MobilenetV1/Conv2d_8_pointwise/weights/read:02CMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_8_pointwise/biases:0,MobilenetV1/Conv2d_8_pointwise/biases/Assign,MobilenetV1/Conv2d_8_pointwise/biases/read:029MobilenetV1/Conv2d_8_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_9_depthwise/depthwise_weights:07MobilenetV1/Conv2d_9_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_9_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_9_depthwise/biases:0,MobilenetV1/Conv2d_9_depthwise/biases/Assign,MobilenetV1/Conv2d_9_depthwise/biases/read:029MobilenetV1/Conv2d_9_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_9_pointwise/weights:0-MobilenetV1/Conv2d_9_pointwise/weights/Assign-MobilenetV1/Conv2d_9_pointwise/weights/read:02CMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_9_pointwise/biases:0,MobilenetV1/Conv2d_9_pointwise/biases/Assign,MobilenetV1/Conv2d_9_pointwise/biases/read:029MobilenetV1/Conv2d_9_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_10_depthwise/depthwise_weights:08MobilenetV1/Conv2d_10_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_10_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_10_depthwise/biases:0-MobilenetV1/Conv2d_10_depthwise/biases/Assign-MobilenetV1/Conv2d_10_depthwise/biases/read:02:MobilenetV1/Conv2d_10_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_10_pointwise/weights:0.MobilenetV1/Conv2d_10_pointwise/weights/Assign.MobilenetV1/Conv2d_10_pointwise/weights/read:02DMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_10_pointwise/biases:0-MobilenetV1/Conv2d_10_pointwise/biases/Assign-MobilenetV1/Conv2d_10_pointwise/biases/read:02:MobilenetV1/Conv2d_10_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_11_depthwise/depthwise_weights:08MobilenetV1/Conv2d_11_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_11_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_11_depthwise/biases:0-MobilenetV1/Conv2d_11_depthwise/biases/Assign-MobilenetV1/Conv2d_11_depthwise/biases/read:02:MobilenetV1/Conv2d_11_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_11_pointwise/weights:0.MobilenetV1/Conv2d_11_pointwise/weights/Assign.MobilenetV1/Conv2d_11_pointwise/weights/read:02DMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_11_pointwise/biases:0-MobilenetV1/Conv2d_11_pointwise/biases/Assign-MobilenetV1/Conv2d_11_pointwise/biases/read:02:MobilenetV1/Conv2d_11_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_12_depthwise/depthwise_weights:08MobilenetV1/Conv2d_12_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_12_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_12_depthwise/biases:0-MobilenetV1/Conv2d_12_depthwise/biases/Assign-MobilenetV1/Conv2d_12_depthwise/biases/read:02:MobilenetV1/Conv2d_12_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_12_pointwise/weights:0.MobilenetV1/Conv2d_12_pointwise/weights/Assign.MobilenetV1/Conv2d_12_pointwise/weights/read:02DMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_12_pointwise/biases:0-MobilenetV1/Conv2d_12_pointwise/biases/Assign-MobilenetV1/Conv2d_12_pointwise/biases/read:02:MobilenetV1/Conv2d_12_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_13_depthwise/depthwise_weights:08MobilenetV1/Conv2d_13_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_13_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_13_depthwise/biases:0-MobilenetV1/Conv2d_13_depthwise/biases/Assign-MobilenetV1/Conv2d_13_depthwise/biases/read:02:MobilenetV1/Conv2d_13_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_13_pointwise/weights:0.MobilenetV1/Conv2d_13_pointwise/weights/Assign.MobilenetV1/Conv2d_13_pointwise/weights/read:02DMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_13_pointwise/biases:0-MobilenetV1/Conv2d_13_pointwise/biases/Assign-MobilenetV1/Conv2d_13_pointwise/biases/read:02:MobilenetV1/Conv2d_13_pointwise/biases/Initializer/zeros:08
?
(MobilenetV1/Embs/Conv2d_1c_1x1/weights:0-MobilenetV1/Embs/Conv2d_1c_1x1/weights/Assign-MobilenetV1/Embs/Conv2d_1c_1x1/weights/read:02CMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform:08
?
'MobilenetV1/Embs/Conv2d_1c_1x1/biases:0,MobilenetV1/Embs/Conv2d_1c_1x1/biases/Assign,MobilenetV1/Embs/Conv2d_1c_1x1/biases/read:029MobilenetV1/Embs/Conv2d_1c_1x1/biases/Initializer/zeros:08"?]
model_variables?\?\
?
MobilenetV1/Conv2d_0/weights:0#MobilenetV1/Conv2d_0/weights/Assign#MobilenetV1/Conv2d_0/weights/read:029MobilenetV1/Conv2d_0/weights/Initializer/random_uniform:08
?
MobilenetV1/Conv2d_0/biases:0"MobilenetV1/Conv2d_0/biases/Assign"MobilenetV1/Conv2d_0/biases/read:02/MobilenetV1/Conv2d_0/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_1_depthwise/depthwise_weights:07MobilenetV1/Conv2d_1_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_1_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_1_depthwise/biases:0,MobilenetV1/Conv2d_1_depthwise/biases/Assign,MobilenetV1/Conv2d_1_depthwise/biases/read:029MobilenetV1/Conv2d_1_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_1_pointwise/weights:0-MobilenetV1/Conv2d_1_pointwise/weights/Assign-MobilenetV1/Conv2d_1_pointwise/weights/read:02CMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_1_pointwise/biases:0,MobilenetV1/Conv2d_1_pointwise/biases/Assign,MobilenetV1/Conv2d_1_pointwise/biases/read:029MobilenetV1/Conv2d_1_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_2_depthwise/depthwise_weights:07MobilenetV1/Conv2d_2_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_2_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_2_depthwise/biases:0,MobilenetV1/Conv2d_2_depthwise/biases/Assign,MobilenetV1/Conv2d_2_depthwise/biases/read:029MobilenetV1/Conv2d_2_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_2_pointwise/weights:0-MobilenetV1/Conv2d_2_pointwise/weights/Assign-MobilenetV1/Conv2d_2_pointwise/weights/read:02CMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_2_pointwise/biases:0,MobilenetV1/Conv2d_2_pointwise/biases/Assign,MobilenetV1/Conv2d_2_pointwise/biases/read:029MobilenetV1/Conv2d_2_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_3_depthwise/depthwise_weights:07MobilenetV1/Conv2d_3_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_3_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_3_depthwise/biases:0,MobilenetV1/Conv2d_3_depthwise/biases/Assign,MobilenetV1/Conv2d_3_depthwise/biases/read:029MobilenetV1/Conv2d_3_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_3_pointwise/weights:0-MobilenetV1/Conv2d_3_pointwise/weights/Assign-MobilenetV1/Conv2d_3_pointwise/weights/read:02CMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_3_pointwise/biases:0,MobilenetV1/Conv2d_3_pointwise/biases/Assign,MobilenetV1/Conv2d_3_pointwise/biases/read:029MobilenetV1/Conv2d_3_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_4_depthwise/depthwise_weights:07MobilenetV1/Conv2d_4_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_4_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_4_depthwise/biases:0,MobilenetV1/Conv2d_4_depthwise/biases/Assign,MobilenetV1/Conv2d_4_depthwise/biases/read:029MobilenetV1/Conv2d_4_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_4_pointwise/weights:0-MobilenetV1/Conv2d_4_pointwise/weights/Assign-MobilenetV1/Conv2d_4_pointwise/weights/read:02CMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_4_pointwise/biases:0,MobilenetV1/Conv2d_4_pointwise/biases/Assign,MobilenetV1/Conv2d_4_pointwise/biases/read:029MobilenetV1/Conv2d_4_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_5_depthwise/depthwise_weights:07MobilenetV1/Conv2d_5_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_5_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_5_depthwise/biases:0,MobilenetV1/Conv2d_5_depthwise/biases/Assign,MobilenetV1/Conv2d_5_depthwise/biases/read:029MobilenetV1/Conv2d_5_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_5_pointwise/weights:0-MobilenetV1/Conv2d_5_pointwise/weights/Assign-MobilenetV1/Conv2d_5_pointwise/weights/read:02CMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_5_pointwise/biases:0,MobilenetV1/Conv2d_5_pointwise/biases/Assign,MobilenetV1/Conv2d_5_pointwise/biases/read:029MobilenetV1/Conv2d_5_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_6_depthwise/depthwise_weights:07MobilenetV1/Conv2d_6_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_6_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_6_depthwise/biases:0,MobilenetV1/Conv2d_6_depthwise/biases/Assign,MobilenetV1/Conv2d_6_depthwise/biases/read:029MobilenetV1/Conv2d_6_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_6_pointwise/weights:0-MobilenetV1/Conv2d_6_pointwise/weights/Assign-MobilenetV1/Conv2d_6_pointwise/weights/read:02CMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_6_pointwise/biases:0,MobilenetV1/Conv2d_6_pointwise/biases/Assign,MobilenetV1/Conv2d_6_pointwise/biases/read:029MobilenetV1/Conv2d_6_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_7_depthwise/depthwise_weights:07MobilenetV1/Conv2d_7_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_7_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_7_depthwise/biases:0,MobilenetV1/Conv2d_7_depthwise/biases/Assign,MobilenetV1/Conv2d_7_depthwise/biases/read:029MobilenetV1/Conv2d_7_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_7_pointwise/weights:0-MobilenetV1/Conv2d_7_pointwise/weights/Assign-MobilenetV1/Conv2d_7_pointwise/weights/read:02CMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_7_pointwise/biases:0,MobilenetV1/Conv2d_7_pointwise/biases/Assign,MobilenetV1/Conv2d_7_pointwise/biases/read:029MobilenetV1/Conv2d_7_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_8_depthwise/depthwise_weights:07MobilenetV1/Conv2d_8_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_8_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_8_depthwise/biases:0,MobilenetV1/Conv2d_8_depthwise/biases/Assign,MobilenetV1/Conv2d_8_depthwise/biases/read:029MobilenetV1/Conv2d_8_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_8_pointwise/weights:0-MobilenetV1/Conv2d_8_pointwise/weights/Assign-MobilenetV1/Conv2d_8_pointwise/weights/read:02CMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_8_pointwise/biases:0,MobilenetV1/Conv2d_8_pointwise/biases/Assign,MobilenetV1/Conv2d_8_pointwise/biases/read:029MobilenetV1/Conv2d_8_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_9_depthwise/depthwise_weights:07MobilenetV1/Conv2d_9_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_9_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_9_depthwise/biases:0,MobilenetV1/Conv2d_9_depthwise/biases/Assign,MobilenetV1/Conv2d_9_depthwise/biases/read:029MobilenetV1/Conv2d_9_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_9_pointwise/weights:0-MobilenetV1/Conv2d_9_pointwise/weights/Assign-MobilenetV1/Conv2d_9_pointwise/weights/read:02CMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_9_pointwise/biases:0,MobilenetV1/Conv2d_9_pointwise/biases/Assign,MobilenetV1/Conv2d_9_pointwise/biases/read:029MobilenetV1/Conv2d_9_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_10_depthwise/depthwise_weights:08MobilenetV1/Conv2d_10_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_10_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_10_depthwise/biases:0-MobilenetV1/Conv2d_10_depthwise/biases/Assign-MobilenetV1/Conv2d_10_depthwise/biases/read:02:MobilenetV1/Conv2d_10_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_10_pointwise/weights:0.MobilenetV1/Conv2d_10_pointwise/weights/Assign.MobilenetV1/Conv2d_10_pointwise/weights/read:02DMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_10_pointwise/biases:0-MobilenetV1/Conv2d_10_pointwise/biases/Assign-MobilenetV1/Conv2d_10_pointwise/biases/read:02:MobilenetV1/Conv2d_10_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_11_depthwise/depthwise_weights:08MobilenetV1/Conv2d_11_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_11_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_11_depthwise/biases:0-MobilenetV1/Conv2d_11_depthwise/biases/Assign-MobilenetV1/Conv2d_11_depthwise/biases/read:02:MobilenetV1/Conv2d_11_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_11_pointwise/weights:0.MobilenetV1/Conv2d_11_pointwise/weights/Assign.MobilenetV1/Conv2d_11_pointwise/weights/read:02DMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_11_pointwise/biases:0-MobilenetV1/Conv2d_11_pointwise/biases/Assign-MobilenetV1/Conv2d_11_pointwise/biases/read:02:MobilenetV1/Conv2d_11_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_12_depthwise/depthwise_weights:08MobilenetV1/Conv2d_12_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_12_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_12_depthwise/biases:0-MobilenetV1/Conv2d_12_depthwise/biases/Assign-MobilenetV1/Conv2d_12_depthwise/biases/read:02:MobilenetV1/Conv2d_12_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_12_pointwise/weights:0.MobilenetV1/Conv2d_12_pointwise/weights/Assign.MobilenetV1/Conv2d_12_pointwise/weights/read:02DMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_12_pointwise/biases:0-MobilenetV1/Conv2d_12_pointwise/biases/Assign-MobilenetV1/Conv2d_12_pointwise/biases/read:02:MobilenetV1/Conv2d_12_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_13_depthwise/depthwise_weights:08MobilenetV1/Conv2d_13_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_13_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_13_depthwise/biases:0-MobilenetV1/Conv2d_13_depthwise/biases/Assign-MobilenetV1/Conv2d_13_depthwise/biases/read:02:MobilenetV1/Conv2d_13_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_13_pointwise/weights:0.MobilenetV1/Conv2d_13_pointwise/weights/Assign.MobilenetV1/Conv2d_13_pointwise/weights/read:02DMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_13_pointwise/biases:0-MobilenetV1/Conv2d_13_pointwise/biases/Assign-MobilenetV1/Conv2d_13_pointwise/biases/read:02:MobilenetV1/Conv2d_13_pointwise/biases/Initializer/zeros:08
?
(MobilenetV1/Embs/Conv2d_1c_1x1/weights:0-MobilenetV1/Embs/Conv2d_1c_1x1/weights/Assign-MobilenetV1/Embs/Conv2d_1c_1x1/weights/read:02CMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform:08
?
'MobilenetV1/Embs/Conv2d_1c_1x1/biases:0,MobilenetV1/Embs/Conv2d_1c_1x1/biases/Assign,MobilenetV1/Embs/Conv2d_1c_1x1/biases/read:029MobilenetV1/Embs/Conv2d_1c_1x1/biases/Initializer/zeros:08"?]
trainable_variables?\?\
?
MobilenetV1/Conv2d_0/weights:0#MobilenetV1/Conv2d_0/weights/Assign#MobilenetV1/Conv2d_0/weights/read:029MobilenetV1/Conv2d_0/weights/Initializer/random_uniform:08
?
MobilenetV1/Conv2d_0/biases:0"MobilenetV1/Conv2d_0/biases/Assign"MobilenetV1/Conv2d_0/biases/read:02/MobilenetV1/Conv2d_0/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_1_depthwise/depthwise_weights:07MobilenetV1/Conv2d_1_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_1_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_1_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_1_depthwise/biases:0,MobilenetV1/Conv2d_1_depthwise/biases/Assign,MobilenetV1/Conv2d_1_depthwise/biases/read:029MobilenetV1/Conv2d_1_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_1_pointwise/weights:0-MobilenetV1/Conv2d_1_pointwise/weights/Assign-MobilenetV1/Conv2d_1_pointwise/weights/read:02CMobilenetV1/Conv2d_1_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_1_pointwise/biases:0,MobilenetV1/Conv2d_1_pointwise/biases/Assign,MobilenetV1/Conv2d_1_pointwise/biases/read:029MobilenetV1/Conv2d_1_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_2_depthwise/depthwise_weights:07MobilenetV1/Conv2d_2_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_2_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_2_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_2_depthwise/biases:0,MobilenetV1/Conv2d_2_depthwise/biases/Assign,MobilenetV1/Conv2d_2_depthwise/biases/read:029MobilenetV1/Conv2d_2_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_2_pointwise/weights:0-MobilenetV1/Conv2d_2_pointwise/weights/Assign-MobilenetV1/Conv2d_2_pointwise/weights/read:02CMobilenetV1/Conv2d_2_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_2_pointwise/biases:0,MobilenetV1/Conv2d_2_pointwise/biases/Assign,MobilenetV1/Conv2d_2_pointwise/biases/read:029MobilenetV1/Conv2d_2_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_3_depthwise/depthwise_weights:07MobilenetV1/Conv2d_3_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_3_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_3_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_3_depthwise/biases:0,MobilenetV1/Conv2d_3_depthwise/biases/Assign,MobilenetV1/Conv2d_3_depthwise/biases/read:029MobilenetV1/Conv2d_3_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_3_pointwise/weights:0-MobilenetV1/Conv2d_3_pointwise/weights/Assign-MobilenetV1/Conv2d_3_pointwise/weights/read:02CMobilenetV1/Conv2d_3_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_3_pointwise/biases:0,MobilenetV1/Conv2d_3_pointwise/biases/Assign,MobilenetV1/Conv2d_3_pointwise/biases/read:029MobilenetV1/Conv2d_3_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_4_depthwise/depthwise_weights:07MobilenetV1/Conv2d_4_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_4_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_4_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_4_depthwise/biases:0,MobilenetV1/Conv2d_4_depthwise/biases/Assign,MobilenetV1/Conv2d_4_depthwise/biases/read:029MobilenetV1/Conv2d_4_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_4_pointwise/weights:0-MobilenetV1/Conv2d_4_pointwise/weights/Assign-MobilenetV1/Conv2d_4_pointwise/weights/read:02CMobilenetV1/Conv2d_4_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_4_pointwise/biases:0,MobilenetV1/Conv2d_4_pointwise/biases/Assign,MobilenetV1/Conv2d_4_pointwise/biases/read:029MobilenetV1/Conv2d_4_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_5_depthwise/depthwise_weights:07MobilenetV1/Conv2d_5_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_5_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_5_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_5_depthwise/biases:0,MobilenetV1/Conv2d_5_depthwise/biases/Assign,MobilenetV1/Conv2d_5_depthwise/biases/read:029MobilenetV1/Conv2d_5_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_5_pointwise/weights:0-MobilenetV1/Conv2d_5_pointwise/weights/Assign-MobilenetV1/Conv2d_5_pointwise/weights/read:02CMobilenetV1/Conv2d_5_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_5_pointwise/biases:0,MobilenetV1/Conv2d_5_pointwise/biases/Assign,MobilenetV1/Conv2d_5_pointwise/biases/read:029MobilenetV1/Conv2d_5_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_6_depthwise/depthwise_weights:07MobilenetV1/Conv2d_6_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_6_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_6_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_6_depthwise/biases:0,MobilenetV1/Conv2d_6_depthwise/biases/Assign,MobilenetV1/Conv2d_6_depthwise/biases/read:029MobilenetV1/Conv2d_6_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_6_pointwise/weights:0-MobilenetV1/Conv2d_6_pointwise/weights/Assign-MobilenetV1/Conv2d_6_pointwise/weights/read:02CMobilenetV1/Conv2d_6_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_6_pointwise/biases:0,MobilenetV1/Conv2d_6_pointwise/biases/Assign,MobilenetV1/Conv2d_6_pointwise/biases/read:029MobilenetV1/Conv2d_6_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_7_depthwise/depthwise_weights:07MobilenetV1/Conv2d_7_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_7_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_7_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_7_depthwise/biases:0,MobilenetV1/Conv2d_7_depthwise/biases/Assign,MobilenetV1/Conv2d_7_depthwise/biases/read:029MobilenetV1/Conv2d_7_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_7_pointwise/weights:0-MobilenetV1/Conv2d_7_pointwise/weights/Assign-MobilenetV1/Conv2d_7_pointwise/weights/read:02CMobilenetV1/Conv2d_7_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_7_pointwise/biases:0,MobilenetV1/Conv2d_7_pointwise/biases/Assign,MobilenetV1/Conv2d_7_pointwise/biases/read:029MobilenetV1/Conv2d_7_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_8_depthwise/depthwise_weights:07MobilenetV1/Conv2d_8_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_8_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_8_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_8_depthwise/biases:0,MobilenetV1/Conv2d_8_depthwise/biases/Assign,MobilenetV1/Conv2d_8_depthwise/biases/read:029MobilenetV1/Conv2d_8_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_8_pointwise/weights:0-MobilenetV1/Conv2d_8_pointwise/weights/Assign-MobilenetV1/Conv2d_8_pointwise/weights/read:02CMobilenetV1/Conv2d_8_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_8_pointwise/biases:0,MobilenetV1/Conv2d_8_pointwise/biases/Assign,MobilenetV1/Conv2d_8_pointwise/biases/read:029MobilenetV1/Conv2d_8_pointwise/biases/Initializer/zeros:08
?
2MobilenetV1/Conv2d_9_depthwise/depthwise_weights:07MobilenetV1/Conv2d_9_depthwise/depthwise_weights/Assign7MobilenetV1/Conv2d_9_depthwise/depthwise_weights/read:02MMobilenetV1/Conv2d_9_depthwise/depthwise_weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_9_depthwise/biases:0,MobilenetV1/Conv2d_9_depthwise/biases/Assign,MobilenetV1/Conv2d_9_depthwise/biases/read:029MobilenetV1/Conv2d_9_depthwise/biases/Initializer/zeros:08
?
(MobilenetV1/Conv2d_9_pointwise/weights:0-MobilenetV1/Conv2d_9_pointwise/weights/Assign-MobilenetV1/Conv2d_9_pointwise/weights/read:02CMobilenetV1/Conv2d_9_pointwise/weights/Initializer/random_uniform:08
?
'MobilenetV1/Conv2d_9_pointwise/biases:0,MobilenetV1/Conv2d_9_pointwise/biases/Assign,MobilenetV1/Conv2d_9_pointwise/biases/read:029MobilenetV1/Conv2d_9_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_10_depthwise/depthwise_weights:08MobilenetV1/Conv2d_10_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_10_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_10_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_10_depthwise/biases:0-MobilenetV1/Conv2d_10_depthwise/biases/Assign-MobilenetV1/Conv2d_10_depthwise/biases/read:02:MobilenetV1/Conv2d_10_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_10_pointwise/weights:0.MobilenetV1/Conv2d_10_pointwise/weights/Assign.MobilenetV1/Conv2d_10_pointwise/weights/read:02DMobilenetV1/Conv2d_10_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_10_pointwise/biases:0-MobilenetV1/Conv2d_10_pointwise/biases/Assign-MobilenetV1/Conv2d_10_pointwise/biases/read:02:MobilenetV1/Conv2d_10_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_11_depthwise/depthwise_weights:08MobilenetV1/Conv2d_11_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_11_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_11_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_11_depthwise/biases:0-MobilenetV1/Conv2d_11_depthwise/biases/Assign-MobilenetV1/Conv2d_11_depthwise/biases/read:02:MobilenetV1/Conv2d_11_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_11_pointwise/weights:0.MobilenetV1/Conv2d_11_pointwise/weights/Assign.MobilenetV1/Conv2d_11_pointwise/weights/read:02DMobilenetV1/Conv2d_11_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_11_pointwise/biases:0-MobilenetV1/Conv2d_11_pointwise/biases/Assign-MobilenetV1/Conv2d_11_pointwise/biases/read:02:MobilenetV1/Conv2d_11_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_12_depthwise/depthwise_weights:08MobilenetV1/Conv2d_12_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_12_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_12_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_12_depthwise/biases:0-MobilenetV1/Conv2d_12_depthwise/biases/Assign-MobilenetV1/Conv2d_12_depthwise/biases/read:02:MobilenetV1/Conv2d_12_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_12_pointwise/weights:0.MobilenetV1/Conv2d_12_pointwise/weights/Assign.MobilenetV1/Conv2d_12_pointwise/weights/read:02DMobilenetV1/Conv2d_12_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_12_pointwise/biases:0-MobilenetV1/Conv2d_12_pointwise/biases/Assign-MobilenetV1/Conv2d_12_pointwise/biases/read:02:MobilenetV1/Conv2d_12_pointwise/biases/Initializer/zeros:08
?
3MobilenetV1/Conv2d_13_depthwise/depthwise_weights:08MobilenetV1/Conv2d_13_depthwise/depthwise_weights/Assign8MobilenetV1/Conv2d_13_depthwise/depthwise_weights/read:02NMobilenetV1/Conv2d_13_depthwise/depthwise_weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_13_depthwise/biases:0-MobilenetV1/Conv2d_13_depthwise/biases/Assign-MobilenetV1/Conv2d_13_depthwise/biases/read:02:MobilenetV1/Conv2d_13_depthwise/biases/Initializer/zeros:08
?
)MobilenetV1/Conv2d_13_pointwise/weights:0.MobilenetV1/Conv2d_13_pointwise/weights/Assign.MobilenetV1/Conv2d_13_pointwise/weights/read:02DMobilenetV1/Conv2d_13_pointwise/weights/Initializer/random_uniform:08
?
(MobilenetV1/Conv2d_13_pointwise/biases:0-MobilenetV1/Conv2d_13_pointwise/biases/Assign-MobilenetV1/Conv2d_13_pointwise/biases/read:02:MobilenetV1/Conv2d_13_pointwise/biases/Initializer/zeros:08
?
(MobilenetV1/Embs/Conv2d_1c_1x1/weights:0-MobilenetV1/Embs/Conv2d_1c_1x1/weights/Assign-MobilenetV1/Embs/Conv2d_1c_1x1/weights/read:02CMobilenetV1/Embs/Conv2d_1c_1x1/weights/Initializer/random_uniform:08
?
'MobilenetV1/Embs/Conv2d_1c_1x1/biases:0,MobilenetV1/Embs/Conv2d_1c_1x1/biases/Assign,MobilenetV1/Embs/Conv2d_1c_1x1/biases/read:029MobilenetV1/Embs/Conv2d_1c_1x1/biases/Initializer/zeros:08*y
serving_defaultf
#
input
Reshape_1:0	?#
output
Reshape_3:02tensorflow/serving/predict