       £K"	  @Qn]÷Abrain.Event:2Ц†≤&сщ     	MЇ†	л8gQn]÷A"ду
]
PlaceholderPlaceholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
_
Placeholder_1Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

d
random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
W
random_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ю
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*
_output_shapes

:@2*
seed2 *
dtype0*
T0*

seed 
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes

:@2
d
random_normalAddrandom_normal/mulrandom_normal/mean*
T0*
_output_shapes

:@2
|
Variable
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
°
Variable/AssignAssignVariablerandom_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable
i
Variable/readIdentityVariable*
T0*
_output_shapes

:@2*
_class
loc:@Variable
Z
zerosConst*
dtype0*
_output_shapes

:2*
valueB2*    
J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
A
addAddzerosadd/y*
_output_shapes

:2*
T0
~

Variable_1
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Э
Variable_1/AssignAssign
Variable_1add*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_1
o
Variable_1/readIdentity
Variable_1*
_output_shapes

:2*
_class
loc:@Variable_1*
T0
Д
MatMulMatMulPlaceholderVariable/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
W
add_1AddMatMulVariable_1/read*'
_output_shapes
:€€€€€€€€€2*
T0
_
Placeholder_2Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
_
Placeholder_3Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
f
random_normal_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Y
random_normal_1/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_1/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
T0*
_output_shapes

:@2
j
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
_output_shapes

:@2*
T0
~

Variable_2
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
©
Variable_2/AssignAssign
Variable_2random_normal_1*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_2*
T0*
use_locking(
o
Variable_2/readIdentity
Variable_2*
T0*
_output_shapes

:@2*
_class
loc:@Variable_2
\
zeros_1Const*
_output_shapes

:2*
dtype0*
valueB2*    
L
add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
G
add_2Addzeros_1add_2/y*
T0*
_output_shapes

:2
~

Variable_3
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Я
Variable_3/AssignAssign
Variable_3add_2*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_3*
T0*
use_locking(
o
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes

:2*
_class
loc:@Variable_3
К
MatMul_1MatMulPlaceholder_2Variable_2/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Y
add_3AddMatMul_1Variable_3/read*
T0*'
_output_shapes
:€€€€€€€€€2
V
dropout/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
E
TanhTanhadd_3*'
_output_shapes
:€€€€€€€€€2*
T0
_
Placeholder_4Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
_
Placeholder_5Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Y
random_normal_2/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_2/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ґ
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes

:@2*
T0
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes

:@2*
T0
~

Variable_4
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
©
Variable_4/AssignAssign
Variable_4random_normal_2*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_4*
T0*
use_locking(
o
Variable_4/readIdentity
Variable_4*
_output_shapes

:@2*
_class
loc:@Variable_4*
T0
\
zeros_2Const*
_output_shapes

:2*
dtype0*
valueB2*    
L
add_4/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
G
add_4Addzeros_2add_4/y*
T0*
_output_shapes

:2
~

Variable_5
VariableV2*
_output_shapes

:2*
	container *
dtype0*
shared_name *
shape
:2
Я
Variable_5/AssignAssign
Variable_5add_4*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_5*
T0*
use_locking(
o
Variable_5/readIdentity
Variable_5*
_output_shapes

:2*
_class
loc:@Variable_5*
T0
К
MatMul_2MatMulPlaceholder_4Variable_4/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Y
add_5AddMatMul_2Variable_5/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_1/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
G
Tanh_1Tanhadd_5*'
_output_shapes
:€€€€€€€€€2*
T0
_
Placeholder_6Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
_
Placeholder_7Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Y
random_normal_3/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
_output_shapes

:@2*
seed2 *
dtype0*
T0*

seed 
Б
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes

:@2*
T0
j
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
T0*
_output_shapes

:@2
~

Variable_6
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
©
Variable_6/AssignAssign
Variable_6random_normal_3*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_6*
T0*
use_locking(
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:@2*
_class
loc:@Variable_6*
T0
\
zeros_3Const*
valueB2*    *
_output_shapes

:2*
dtype0
L
add_6/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
G
add_6Addzeros_3add_6/y*
T0*
_output_shapes

:2
~

Variable_7
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Я
Variable_7/AssignAssign
Variable_7add_6*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_7
o
Variable_7/readIdentity
Variable_7*
T0*
_output_shapes

:2*
_class
loc:@Variable_7
К
MatMul_3MatMulPlaceholder_6Variable_6/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Y
add_7AddMatMul_3Variable_7/read*
T0*'
_output_shapes
:€€€€€€€€€2
_
Placeholder_8Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
_
Placeholder_9Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

f
random_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Y
random_normal_4/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_4/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ґ
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes

:@2*
T0
j
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
_output_shapes

:@2*
T0
~

Variable_8
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
©
Variable_8/AssignAssign
Variable_8random_normal_4*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_8*
T0*
use_locking(
o
Variable_8/readIdentity
Variable_8*
_class
loc:@Variable_8*
_output_shapes

:@2*
T0
\
zeros_4Const*
valueB2*    *
_output_shapes

:2*
dtype0
L
add_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
G
add_8Addzeros_4add_8/y*
_output_shapes

:2*
T0
~

Variable_9
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Я
Variable_9/AssignAssign
Variable_9add_8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_9
o
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9*
_output_shapes

:2
К
MatMul_4MatMulPlaceholder_8Variable_8/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Y
add_9AddMatMul_4Variable_9/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_2/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
G
Tanh_2Tanhadd_9*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_10Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_11Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_5/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Y
random_normal_5/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_5/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ґ
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Б
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
T0*
_output_shapes

:@2
j
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
T0*
_output_shapes

:@2

Variable_10
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
ђ
Variable_10/AssignAssignVariable_10random_normal_5*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_10*
T0*
use_locking(
r
Variable_10/readIdentityVariable_10*
_class
loc:@Variable_10*
_output_shapes

:@2*
T0
\
zeros_5Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_10/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
I
add_10Addzeros_5add_10/y*
_output_shapes

:2*
T0

Variable_11
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_11/AssignAssignVariable_11add_10*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_11*
T0*
use_locking(
r
Variable_11/readIdentityVariable_11*
T0*
_output_shapes

:2*
_class
loc:@Variable_11
М
MatMul_5MatMulPlaceholder_10Variable_10/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_11AddMatMul_5Variable_11/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_3/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
H
Tanh_3Tanhadd_11*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_12Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_13Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

f
random_normal_6/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Y
random_normal_6/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_6/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ґ
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes

:@2
j
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
T0*
_output_shapes

:@2

Variable_12
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
ђ
Variable_12/AssignAssignVariable_12random_normal_6*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_12*
T0*
use_locking(
r
Variable_12/readIdentityVariable_12*
_class
loc:@Variable_12*
_output_shapes

:@2*
T0
\
zeros_6Const*
valueB2*    *
_output_shapes

:2*
dtype0
M
add_12/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
I
add_12Addzeros_6add_12/y*
_output_shapes

:2*
T0

Variable_13
VariableV2*
_output_shapes

:2*
	container *
dtype0*
shared_name *
shape
:2
£
Variable_13/AssignAssignVariable_13add_12*
_class
loc:@Variable_13*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
r
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13*
_output_shapes

:2
М
MatMul_6MatMulPlaceholder_12Variable_12/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_13AddMatMul_6Variable_13/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_4/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
H
Tanh_4Tanhadd_13*'
_output_shapes
:€€€€€€€€€2*
T0
`
Placeholder_14Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_15Placeholder*'
_output_shapes
:€€€€€€€€€
*
shape: *
dtype0
f
random_normal_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Y
random_normal_7/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_7/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ґ
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Б
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes

:@2*
T0
j
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes

:@2*
T0

Variable_14
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
ђ
Variable_14/AssignAssignVariable_14random_normal_7*
use_locking(*
T0*
_class
loc:@Variable_14*
validate_shape(*
_output_shapes

:@2
r
Variable_14/readIdentityVariable_14*
_class
loc:@Variable_14*
_output_shapes

:@2*
T0
\
zeros_7Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_14/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
I
add_14Addzeros_7add_14/y*
T0*
_output_shapes

:2

Variable_15
VariableV2*
_output_shapes

:2*
	container *
dtype0*
shared_name *
shape
:2
£
Variable_15/AssignAssignVariable_15add_14*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes

:2
r
Variable_15/readIdentityVariable_15*
_output_shapes

:2*
_class
loc:@Variable_15*
T0
М
MatMul_7MatMulPlaceholder_14Variable_14/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_15AddMatMul_7Variable_15/read*'
_output_shapes
:€€€€€€€€€2*
T0
X
dropout_5/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
H
Tanh_5Tanhadd_15*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_16Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_17Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

f
random_normal_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Y
random_normal_8/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_8/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Б
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
_output_shapes

:@2*
T0
j
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes

:@2*
T0

Variable_16
VariableV2*
_output_shapes

:@2*
	container *
shape
:@2*
dtype0*
shared_name 
ђ
Variable_16/AssignAssignVariable_16random_normal_8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_16
r
Variable_16/readIdentityVariable_16*
_class
loc:@Variable_16*
_output_shapes

:@2*
T0
\
zeros_8Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_16/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
I
add_16Addzeros_8add_16/y*
T0*
_output_shapes

:2

Variable_17
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_17/AssignAssignVariable_17add_16*
use_locking(*
T0*
_class
loc:@Variable_17*
validate_shape(*
_output_shapes

:2
r
Variable_17/readIdentityVariable_17*
T0*
_output_shapes

:2*
_class
loc:@Variable_17
М
MatMul_8MatMulPlaceholder_16Variable_16/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_17AddMatMul_8Variable_17/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_6/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
H
Tanh_6Tanhadd_17*
T0*'
_output_shapes
:€€€€€€€€€2
f
random_normal_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"2   
   
Y
random_normal_9/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_9/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
Ґ
$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Б
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
_output_shapes

:2
*
T0
j
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0*
_output_shapes

:2


Variable_18
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
ђ
Variable_18/AssignAssignVariable_18random_normal_9*
_class
loc:@Variable_18*
_output_shapes

:2
*
T0*
validate_shape(*
use_locking(
r
Variable_18/readIdentityVariable_18*
T0*
_class
loc:@Variable_18*
_output_shapes

:2

\
zeros_9Const*
_output_shapes

:
*
dtype0*
valueB
*    
M
add_18/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
I
add_18Addzeros_9add_18/y*
_output_shapes

:
*
T0

Variable_19
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
£
Variable_19/AssignAssignVariable_19add_18*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
*
_class
loc:@Variable_19
r
Variable_19/readIdentityVariable_19*
_output_shapes

:
*
_class
loc:@Variable_19*
T0
Д
MatMul_9MatMulTanh_6Variable_18/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
[
add_19AddMatMul_9Variable_19/read*'
_output_shapes
:€€€€€€€€€
*
T0
X
dropout_7/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
L
SoftmaxSoftmaxadd_19*'
_output_shapes
:€€€€€€€€€
*
T0
E
LogLogSoftmax*'
_output_shapes
:€€€€€€€€€
*
T0
Q
mulMulPlaceholder_17Log*'
_output_shapes
:€€€€€€€€€
*
T0
_
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
q
SumSummulSum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
=
NegNegSum*
T0*#
_output_shapes
:€€€€€€€€€
O
ConstConst*
_output_shapes
:*
dtype0*
valueB: 
V
MeanMeanNegConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
`
Placeholder_18Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_19Placeholder*'
_output_shapes
:€€€€€€€€€
*
shape: *
dtype0
g
random_normal_10/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Z
random_normal_10/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_10/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
§
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
_output_shapes

:@2*
T0
m
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
_output_shapes

:@2*
T0

Variable_20
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_20/AssignAssignVariable_20random_normal_10*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_20
r
Variable_20/readIdentityVariable_20*
T0*
_output_shapes

:@2*
_class
loc:@Variable_20
]
zeros_10Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_20/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_20Addzeros_10add_20/y*
_output_shapes

:2*
T0

Variable_21
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_21/AssignAssignVariable_21add_20*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_21
r
Variable_21/readIdentityVariable_21*
_output_shapes

:2*
_class
loc:@Variable_21*
T0
Н
	MatMul_10MatMulPlaceholder_18Variable_20/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_21Add	MatMul_10Variable_21/read*'
_output_shapes
:€€€€€€€€€2*
T0
X
dropout_8/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
H
Tanh_7Tanhadd_21*'
_output_shapes
:€€€€€€€€€2*
T0
g
random_normal_11/shapeConst*
dtype0*
_output_shapes
:*
valueB"2   
   
Z
random_normal_11/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_11/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
T0*
_output_shapes

:2

m
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
T0*
_output_shapes

:2


Variable_22
VariableV2*
_output_shapes

:2
*
	container *
dtype0*
shared_name *
shape
:2

≠
Variable_22/AssignAssignVariable_22random_normal_11*
use_locking(*
T0*
_class
loc:@Variable_22*
validate_shape(*
_output_shapes

:2

r
Variable_22/readIdentityVariable_22*
T0*
_output_shapes

:2
*
_class
loc:@Variable_22
]
zeros_11Const*
valueB
*    *
dtype0*
_output_shapes

:

M
add_22/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_22Addzeros_11add_22/y*
T0*
_output_shapes

:


Variable_23
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
£
Variable_23/AssignAssignVariable_23add_22*
use_locking(*
T0*
_class
loc:@Variable_23*
validate_shape(*
_output_shapes

:

r
Variable_23/readIdentityVariable_23*
_class
loc:@Variable_23*
_output_shapes

:
*
T0
Е
	MatMul_11MatMulTanh_7Variable_22/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
\
add_23Add	MatMul_11Variable_23/read*
T0*'
_output_shapes
:€€€€€€€€€

X
dropout_9/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_1Softmaxadd_23*'
_output_shapes
:€€€€€€€€€
*
T0
I
Log_1Log	Softmax_1*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_1MulPlaceholder_19Log_1*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_1/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
w
Sum_1Summul_1Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_1NegSum_1*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_1MeanNeg_1Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
N
	loss/tagsConst*
valueB
 Bloss*
_output_shapes
: *
dtype0
I
lossScalarSummary	loss/tagsMean_1*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Y
gradients/FillFillgradients/Shapegradients/Const*
_output_shapes
: *
T0
m
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Р
gradients/Mean_1_grad/ReshapeReshapegradients/Fill#gradients/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
`
gradients/Mean_1_grad/ShapeShapeNeg_1*
T0*
out_type0*
_output_shapes
:
Ю
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
b
gradients/Mean_1_grad/Shape_1ShapeNeg_1*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
e
gradients/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ь
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
g
gradients/Mean_1_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
†
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
И
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
Ж
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
_output_shapes
: *
T0
r
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
О
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
l
gradients/Neg_1_grad/NegNeggradients/Mean_1_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
_
gradients/Sum_1_grad/ShapeShapemul_1*
_output_shapes
:*
out_type0*
T0
[
gradients/Sum_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
x
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
T0*
_output_shapes
:
~
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
T0*
_output_shapes
:
f
gradients/Sum_1_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
b
 gradients/Sum_1_grad/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 gradients/Sum_1_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
™
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*
_output_shapes
:*

Tidx0
a
gradients/Sum_1_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Е
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
T0*
_output_shapes
:
„
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
`
gradients/Sum_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Щ
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
И
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
T0
Ц
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_1_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
Ґ
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

h
gradients/mul_1_grad/ShapeShapePlaceholder_19*
T0*
_output_shapes
:*
out_type0
a
gradients/mul_1_grad/Shape_1ShapeLog_1*
T0*
out_type0*
_output_shapes
:
Ї
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
s
gradients/mul_1_grad/mulMulgradients/Sum_1_grad/TileLog_1*'
_output_shapes
:€€€€€€€€€
*
T0
•
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

~
gradients/mul_1_grad/mul_1MulPlaceholder_19gradients/Sum_1_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
в
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*/
_class%
#!loc:@gradients/mul_1_grad/Reshape*
T0
и
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1
Ь
gradients/Log_1_grad/Reciprocal
Reciprocal	Softmax_10^gradients/mul_1_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

£
gradients/Log_1_grad/mulMul/gradients/mul_1_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
z
gradients/Softmax_1_grad/mulMulgradients/Log_1_grad/mul	Softmax_1*'
_output_shapes
:€€€€€€€€€
*
T0
x
.gradients/Softmax_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Љ
gradients/Softmax_1_grad/SumSumgradients/Softmax_1_grad/mul.gradients/Softmax_1_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
w
&gradients/Softmax_1_grad/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
±
 gradients/Softmax_1_grad/ReshapeReshapegradients/Softmax_1_grad/Sum&gradients/Softmax_1_grad/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
С
gradients/Softmax_1_grad/subSubgradients/Log_1_grad/mul gradients/Softmax_1_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

А
gradients/Softmax_1_grad/mul_1Mulgradients/Softmax_1_grad/sub	Softmax_1*
T0*'
_output_shapes
:€€€€€€€€€

d
gradients/add_23_grad/ShapeShape	MatMul_11*
_output_shapes
:*
out_type0*
T0
n
gradients/add_23_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
љ
+gradients/add_23_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_23_grad/Shapegradients/add_23_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≠
gradients/add_23_grad/SumSumgradients/Softmax_1_grad/mul_1+gradients/add_23_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
†
gradients/add_23_grad/ReshapeReshapegradients/add_23_grad/Sumgradients/add_23_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
±
gradients/add_23_grad/Sum_1Sumgradients/Softmax_1_grad/mul_1-gradients/add_23_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
gradients/add_23_grad/Reshape_1Reshapegradients/add_23_grad/Sum_1gradients/add_23_grad/Shape_1*
_output_shapes

:
*
Tshape0*
T0
p
&gradients/add_23_grad/tuple/group_depsNoOp^gradients/add_23_grad/Reshape ^gradients/add_23_grad/Reshape_1
ж
.gradients/add_23_grad/tuple/control_dependencyIdentitygradients/add_23_grad/Reshape'^gradients/add_23_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*0
_class&
$"loc:@gradients/add_23_grad/Reshape
г
0gradients/add_23_grad/tuple/control_dependency_1Identitygradients/add_23_grad/Reshape_1'^gradients/add_23_grad/tuple/group_deps*
T0*
_output_shapes

:
*2
_class(
&$loc:@gradients/add_23_grad/Reshape_1
√
gradients/MatMul_11_grad/MatMulMatMul.gradients/add_23_grad/tuple/control_dependencyVariable_22/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
≤
!gradients/MatMul_11_grad/MatMul_1MatMulTanh_7.gradients/add_23_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
w
)gradients/MatMul_11_grad/tuple/group_depsNoOp ^gradients/MatMul_11_grad/MatMul"^gradients/MatMul_11_grad/MatMul_1
р
1gradients/MatMul_11_grad/tuple/control_dependencyIdentitygradients/MatMul_11_grad/MatMul*^gradients/MatMul_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/MatMul_11_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
н
3gradients/MatMul_11_grad/tuple/control_dependency_1Identity!gradients/MatMul_11_grad/MatMul_1*^gradients/MatMul_11_grad/tuple/group_deps*4
_class*
(&loc:@gradients/MatMul_11_grad/MatMul_1*
_output_shapes

:2
*
T0
Ч
gradients/Tanh_7_grad/TanhGradTanhGradTanh_71gradients/MatMul_11_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
d
gradients/add_21_grad/ShapeShape	MatMul_10*
_output_shapes
:*
out_type0*
T0
n
gradients/add_21_grad/Shape_1Const*
valueB"   2   *
_output_shapes
:*
dtype0
љ
+gradients/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_21_grad/Shapegradients/add_21_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≠
gradients/add_21_grad/SumSumgradients/Tanh_7_grad/TanhGrad+gradients/add_21_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
†
gradients/add_21_grad/ReshapeReshapegradients/add_21_grad/Sumgradients/add_21_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€2*
T0
±
gradients/add_21_grad/Sum_1Sumgradients/Tanh_7_grad/TanhGrad-gradients/add_21_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
gradients/add_21_grad/Reshape_1Reshapegradients/add_21_grad/Sum_1gradients/add_21_grad/Shape_1*
Tshape0*
_output_shapes

:2*
T0
p
&gradients/add_21_grad/tuple/group_depsNoOp^gradients/add_21_grad/Reshape ^gradients/add_21_grad/Reshape_1
ж
.gradients/add_21_grad/tuple/control_dependencyIdentitygradients/add_21_grad/Reshape'^gradients/add_21_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*0
_class&
$"loc:@gradients/add_21_grad/Reshape
г
0gradients/add_21_grad/tuple/control_dependency_1Identitygradients/add_21_grad/Reshape_1'^gradients/add_21_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/add_21_grad/Reshape_1*
_output_shapes

:2
√
gradients/MatMul_10_grad/MatMulMatMul.gradients/add_21_grad/tuple/control_dependencyVariable_20/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
T0
Ї
!gradients/MatMul_10_grad/MatMul_1MatMulPlaceholder_18.gradients/add_21_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
w
)gradients/MatMul_10_grad/tuple/group_depsNoOp ^gradients/MatMul_10_grad/MatMul"^gradients/MatMul_10_grad/MatMul_1
р
1gradients/MatMul_10_grad/tuple/control_dependencyIdentitygradients/MatMul_10_grad/MatMul*^gradients/MatMul_10_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/MatMul_10_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
н
3gradients/MatMul_10_grad/tuple/control_dependency_1Identity!gradients/MatMul_10_grad/MatMul_1*^gradients/MatMul_10_grad/tuple/group_deps*4
_class*
(&loc:@gradients/MatMul_10_grad/MatMul_1*
_output_shapes

:@2*
T0
b
GradientDescent/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
М
7GradientDescent/update_Variable_20/ApplyGradientDescentApplyGradientDescentVariable_20GradientDescent/learning_rate3gradients/MatMul_10_grad/tuple/control_dependency_1*
_class
loc:@Variable_20*
_output_shapes

:@2*
T0*
use_locking( 
Й
7GradientDescent/update_Variable_21/ApplyGradientDescentApplyGradientDescentVariable_21GradientDescent/learning_rate0gradients/add_21_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:2*
_class
loc:@Variable_21
М
7GradientDescent/update_Variable_22/ApplyGradientDescentApplyGradientDescentVariable_22GradientDescent/learning_rate3gradients/MatMul_11_grad/tuple/control_dependency_1*
_class
loc:@Variable_22*
_output_shapes

:2
*
T0*
use_locking( 
Й
7GradientDescent/update_Variable_23/ApplyGradientDescentApplyGradientDescentVariable_23GradientDescent/learning_rate0gradients/add_23_grad/tuple/control_dependency_1*
_output_shapes

:
*
_class
loc:@Variable_23*
T0*
use_locking( 
€
GradientDescentNoOp8^GradientDescent/update_Variable_20/ApplyGradientDescent8^GradientDescent/update_Variable_21/ApplyGradientDescent8^GradientDescent/update_Variable_22/ApplyGradientDescent8^GradientDescent/update_Variable_23/ApplyGradientDescent
Q
Placeholder_20Placeholder*
dtype0*
shape: *
_output_shapes
:
`
Placeholder_21Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
`
Placeholder_22Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

g
random_normal_12/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Z
random_normal_12/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_12/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
§
%random_normal_12/RandomStandardNormalRandomStandardNormalrandom_normal_12/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_12/mulMul%random_normal_12/RandomStandardNormalrandom_normal_12/stddev*
T0*
_output_shapes

:@2
m
random_normal_12Addrandom_normal_12/mulrandom_normal_12/mean*
_output_shapes

:@2*
T0

Variable_24
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_24/AssignAssignVariable_24random_normal_12*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_24
r
Variable_24/readIdentityVariable_24*
T0*
_output_shapes

:@2*
_class
loc:@Variable_24
]
zeros_12Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_24/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_24Addzeros_12add_24/y*
_output_shapes

:2*
T0

Variable_25
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_25/AssignAssignVariable_25add_24*
use_locking(*
T0*
_class
loc:@Variable_25*
validate_shape(*
_output_shapes

:2
r
Variable_25/readIdentityVariable_25*
_output_shapes

:2*
_class
loc:@Variable_25*
T0
Н
	MatMul_12MatMulPlaceholder_21Variable_24/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_25Add	MatMul_12Variable_25/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_10/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
H
Tanh_8Tanhadd_25*
T0*'
_output_shapes
:€€€€€€€€€2
g
random_normal_13/shapeConst*
_output_shapes
:*
dtype0*
valueB"2   
   
Z
random_normal_13/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_13/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_13/RandomStandardNormalRandomStandardNormalrandom_normal_13/shape*
_output_shapes

:2
*
seed2 *
T0*

seed *
dtype0
Д
random_normal_13/mulMul%random_normal_13/RandomStandardNormalrandom_normal_13/stddev*
T0*
_output_shapes

:2

m
random_normal_13Addrandom_normal_13/mulrandom_normal_13/mean*
T0*
_output_shapes

:2


Variable_26
VariableV2*
_output_shapes

:2
*
	container *
shape
:2
*
dtype0*
shared_name 
≠
Variable_26/AssignAssignVariable_26random_normal_13*
_class
loc:@Variable_26*
_output_shapes

:2
*
T0*
validate_shape(*
use_locking(
r
Variable_26/readIdentityVariable_26*
T0*
_class
loc:@Variable_26*
_output_shapes

:2

]
zeros_13Const*
valueB
*    *
_output_shapes

:
*
dtype0
M
add_26/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_26Addzeros_13add_26/y*
_output_shapes

:
*
T0

Variable_27
VariableV2*
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*
	container 
£
Variable_27/AssignAssignVariable_27add_26*
_output_shapes

:
*
validate_shape(*
_class
loc:@Variable_27*
T0*
use_locking(
r
Variable_27/readIdentityVariable_27*
_class
loc:@Variable_27*
_output_shapes

:
*
T0
Е
	MatMul_13MatMulTanh_8Variable_26/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
\
add_27Add	MatMul_13Variable_27/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_11/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
N
	Softmax_2Softmaxadd_27*
T0*'
_output_shapes
:€€€€€€€€€

I
Log_2Log	Softmax_2*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_2MulPlaceholder_22Log_2*
T0*'
_output_shapes
:€€€€€€€€€

a
Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
w
Sum_2Summul_2Sum_2/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_2NegSum_2*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2MeanNeg_2Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
loss_1/tagsConst*
dtype0*
_output_shapes
: *
valueB Bloss_1
M
loss_1ScalarSummaryloss_1/tagsMean_2*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_1/ConstConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*
_output_shapes
: 
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_1/Mean_2_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_2_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients_1/Mean_2_grad/ShapeShapeNeg_2*
T0*
out_type0*
_output_shapes
:
§
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
d
gradients_1/Mean_2_grad/Shape_1ShapeNeg_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_1/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
¶
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
О
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
_output_shapes
: *
T0
М
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_1/Neg_2_grad/NegNeggradients_1/Mean_2_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_1/Sum_2_grad/ShapeShapemul_2*
T0*
out_type0*
_output_shapes
:
]
gradients_1/Sum_2_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
|
gradients_1/Sum_2_grad/addAddSum_2/reduction_indicesgradients_1/Sum_2_grad/Size*
T0*
_output_shapes
:
Д
gradients_1/Sum_2_grad/modFloorModgradients_1/Sum_2_grad/addgradients_1/Sum_2_grad/Size*
T0*
_output_shapes
:
h
gradients_1/Sum_2_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
d
"gradients_1/Sum_2_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"gradients_1/Sum_2_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
≤
gradients_1/Sum_2_grad/rangeRange"gradients_1/Sum_2_grad/range/startgradients_1/Sum_2_grad/Size"gradients_1/Sum_2_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_1/Sum_2_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
Л
gradients_1/Sum_2_grad/FillFillgradients_1/Sum_2_grad/Shape_1!gradients_1/Sum_2_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_1/Sum_2_grad/DynamicStitchDynamicStitchgradients_1/Sum_2_grad/rangegradients_1/Sum_2_grad/modgradients_1/Sum_2_grad/Shapegradients_1/Sum_2_grad/Fill*
N*
T0*#
_output_shapes
:€€€€€€€€€
b
 gradients_1/Sum_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_1/Sum_2_grad/MaximumMaximum$gradients_1/Sum_2_grad/DynamicStitch gradients_1/Sum_2_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
О
gradients_1/Sum_2_grad/floordivFloorDivgradients_1/Sum_2_grad/Shapegradients_1/Sum_2_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_1/Sum_2_grad/ReshapeReshapegradients_1/Neg_2_grad/Neg$gradients_1/Sum_2_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
®
gradients_1/Sum_2_grad/TileTilegradients_1/Sum_2_grad/Reshapegradients_1/Sum_2_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_1/mul_2_grad/ShapeShapePlaceholder_22*
T0*
out_type0*
_output_shapes
:
c
gradients_1/mul_2_grad/Shape_1ShapeLog_2*
T0*
_output_shapes
:*
out_type0
ј
,gradients_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_2_grad/Shapegradients_1/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
w
gradients_1/mul_2_grad/mulMulgradients_1/Sum_2_grad/TileLog_2*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_1/mul_2_grad/SumSumgradients_1/mul_2_grad/mul,gradients_1/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_1/mul_2_grad/ReshapeReshapegradients_1/mul_2_grad/Sumgradients_1/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

В
gradients_1/mul_2_grad/mul_1MulPlaceholder_22gradients_1/Sum_2_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_1/mul_2_grad/Sum_1Sumgradients_1/mul_2_grad/mul_1.gradients_1/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
©
 gradients_1/mul_2_grad/Reshape_1Reshapegradients_1/mul_2_grad/Sum_1gradients_1/mul_2_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
s
'gradients_1/mul_2_grad/tuple/group_depsNoOp^gradients_1/mul_2_grad/Reshape!^gradients_1/mul_2_grad/Reshape_1
к
/gradients_1/mul_2_grad/tuple/control_dependencyIdentitygradients_1/mul_2_grad/Reshape(^gradients_1/mul_2_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_1/mul_2_grad/Reshape
р
1gradients_1/mul_2_grad/tuple/control_dependency_1Identity gradients_1/mul_2_grad/Reshape_1(^gradients_1/mul_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/mul_2_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
*
T0
†
!gradients_1/Log_2_grad/Reciprocal
Reciprocal	Softmax_22^gradients_1/mul_2_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

©
gradients_1/Log_2_grad/mulMul1gradients_1/mul_2_grad/tuple/control_dependency_1!gradients_1/Log_2_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

~
gradients_1/Softmax_2_grad/mulMulgradients_1/Log_2_grad/mul	Softmax_2*'
_output_shapes
:€€€€€€€€€
*
T0
z
0gradients_1/Softmax_2_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
¬
gradients_1/Softmax_2_grad/SumSumgradients_1/Softmax_2_grad/mul0gradients_1/Softmax_2_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_1/Softmax_2_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients_1/Softmax_2_grad/ReshapeReshapegradients_1/Softmax_2_grad/Sum(gradients_1/Softmax_2_grad/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
Ч
gradients_1/Softmax_2_grad/subSubgradients_1/Log_2_grad/mul"gradients_1/Softmax_2_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_1/Softmax_2_grad/mul_1Mulgradients_1/Softmax_2_grad/sub	Softmax_2*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_1/add_27_grad/ShapeShape	MatMul_13*
_output_shapes
:*
out_type0*
T0
p
gradients_1/add_27_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
√
-gradients_1/add_27_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_27_grad/Shapegradients_1/add_27_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_1/add_27_grad/SumSum gradients_1/Softmax_2_grad/mul_1-gradients_1/add_27_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_1/add_27_grad/ReshapeReshapegradients_1/add_27_grad/Sumgradients_1/add_27_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
Ј
gradients_1/add_27_grad/Sum_1Sum gradients_1/Softmax_2_grad/mul_1/gradients_1/add_27_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_1/add_27_grad/Reshape_1Reshapegradients_1/add_27_grad/Sum_1gradients_1/add_27_grad/Shape_1*
T0*
_output_shapes

:
*
Tshape0
v
(gradients_1/add_27_grad/tuple/group_depsNoOp ^gradients_1/add_27_grad/Reshape"^gradients_1/add_27_grad/Reshape_1
о
0gradients_1/add_27_grad/tuple/control_dependencyIdentitygradients_1/add_27_grad/Reshape)^gradients_1/add_27_grad/tuple/group_deps*2
_class(
&$loc:@gradients_1/add_27_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_1/add_27_grad/tuple/control_dependency_1Identity!gradients_1/add_27_grad/Reshape_1)^gradients_1/add_27_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/add_27_grad/Reshape_1*
_output_shapes

:

«
!gradients_1/MatMul_13_grad/MatMulMatMul0gradients_1/add_27_grad/tuple/control_dependencyVariable_26/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
ґ
#gradients_1/MatMul_13_grad/MatMul_1MatMulTanh_80gradients_1/add_27_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_1/MatMul_13_grad/tuple/group_depsNoOp"^gradients_1/MatMul_13_grad/MatMul$^gradients_1/MatMul_13_grad/MatMul_1
ш
3gradients_1/MatMul_13_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_13_grad/MatMul,^gradients_1/MatMul_13_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/MatMul_13_grad/MatMul*'
_output_shapes
:€€€€€€€€€2*
T0
х
5gradients_1/MatMul_13_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_13_grad/MatMul_1,^gradients_1/MatMul_13_grad/tuple/group_deps*
T0*
_output_shapes

:2
*6
_class,
*(loc:@gradients_1/MatMul_13_grad/MatMul_1
Ы
 gradients_1/Tanh_8_grad/TanhGradTanhGradTanh_83gradients_1/MatMul_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_1/add_25_grad/ShapeShape	MatMul_12*
_output_shapes
:*
out_type0*
T0
p
gradients_1/add_25_grad/Shape_1Const*
valueB"   2   *
dtype0*
_output_shapes
:
√
-gradients_1/add_25_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_25_grad/Shapegradients_1/add_25_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_1/add_25_grad/SumSum gradients_1/Tanh_8_grad/TanhGrad-gradients_1/add_25_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_1/add_25_grad/ReshapeReshapegradients_1/add_25_grad/Sumgradients_1/add_25_grad/Shape*'
_output_shapes
:€€€€€€€€€2*
Tshape0*
T0
Ј
gradients_1/add_25_grad/Sum_1Sum gradients_1/Tanh_8_grad/TanhGrad/gradients_1/add_25_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_1/add_25_grad/Reshape_1Reshapegradients_1/add_25_grad/Sum_1gradients_1/add_25_grad/Shape_1*
_output_shapes

:2*
Tshape0*
T0
v
(gradients_1/add_25_grad/tuple/group_depsNoOp ^gradients_1/add_25_grad/Reshape"^gradients_1/add_25_grad/Reshape_1
о
0gradients_1/add_25_grad/tuple/control_dependencyIdentitygradients_1/add_25_grad/Reshape)^gradients_1/add_25_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/add_25_grad/Reshape*'
_output_shapes
:€€€€€€€€€2
л
2gradients_1/add_25_grad/tuple/control_dependency_1Identity!gradients_1/add_25_grad/Reshape_1)^gradients_1/add_25_grad/tuple/group_deps*
_output_shapes

:2*4
_class*
(&loc:@gradients_1/add_25_grad/Reshape_1*
T0
«
!gradients_1/MatMul_12_grad/MatMulMatMul0gradients_1/add_25_grad/tuple/control_dependencyVariable_24/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_1/MatMul_12_grad/MatMul_1MatMulPlaceholder_210gradients_1/add_25_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_1/MatMul_12_grad/tuple/group_depsNoOp"^gradients_1/MatMul_12_grad/MatMul$^gradients_1/MatMul_12_grad/MatMul_1
ш
3gradients_1/MatMul_12_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_12_grad/MatMul,^gradients_1/MatMul_12_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/MatMul_12_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
х
5gradients_1/MatMul_12_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_12_grad/MatMul_1,^gradients_1/MatMul_12_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_12_grad/MatMul_1*
_output_shapes

:@2
d
GradientDescent_1/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
Т
9GradientDescent_1/update_Variable_24/ApplyGradientDescentApplyGradientDescentVariable_24GradientDescent_1/learning_rate5gradients_1/MatMul_12_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:@2*
_class
loc:@Variable_24
П
9GradientDescent_1/update_Variable_25/ApplyGradientDescentApplyGradientDescentVariable_25GradientDescent_1/learning_rate2gradients_1/add_25_grad/tuple/control_dependency_1*
_class
loc:@Variable_25*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_1/update_Variable_26/ApplyGradientDescentApplyGradientDescentVariable_26GradientDescent_1/learning_rate5gradients_1/MatMul_13_grad/tuple/control_dependency_1*
_class
loc:@Variable_26*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_1/update_Variable_27/ApplyGradientDescentApplyGradientDescentVariable_27GradientDescent_1/learning_rate2gradients_1/add_27_grad/tuple/control_dependency_1*
_output_shapes

:
*
_class
loc:@Variable_27*
T0*
use_locking( 
Й
GradientDescent_1NoOp:^GradientDescent_1/update_Variable_24/ApplyGradientDescent:^GradientDescent_1/update_Variable_25/ApplyGradientDescent:^GradientDescent_1/update_Variable_26/ApplyGradientDescent:^GradientDescent_1/update_Variable_27/ApplyGradientDescent
Q
Placeholder_23Placeholder*
_output_shapes
:*
shape: *
dtype0
Q
Merge/MergeSummaryMergeSummarylossloss_1*
N*
_output_shapes
: 
`
Placeholder_24Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_25Placeholder*'
_output_shapes
:€€€€€€€€€
*
shape: *
dtype0
g
random_normal_14/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Z
random_normal_14/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_14/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_14/RandomStandardNormalRandomStandardNormalrandom_normal_14/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_14/mulMul%random_normal_14/RandomStandardNormalrandom_normal_14/stddev*
_output_shapes

:@2*
T0
m
random_normal_14Addrandom_normal_14/mulrandom_normal_14/mean*
_output_shapes

:@2*
T0

Variable_28
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
≠
Variable_28/AssignAssignVariable_28random_normal_14*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_28*
T0*
use_locking(
r
Variable_28/readIdentityVariable_28*
_class
loc:@Variable_28*
_output_shapes

:@2*
T0
]
zeros_14Const*
valueB2*    *
_output_shapes

:2*
dtype0
M
add_28/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
J
add_28Addzeros_14add_28/y*
T0*
_output_shapes

:2

Variable_29
VariableV2*
_output_shapes

:2*
	container *
dtype0*
shared_name *
shape
:2
£
Variable_29/AssignAssignVariable_29add_28*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_29*
T0*
use_locking(
r
Variable_29/readIdentityVariable_29*
_output_shapes

:2*
_class
loc:@Variable_29*
T0
Н
	MatMul_14MatMulPlaceholder_24Variable_28/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_29Add	MatMul_14Variable_29/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_12/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
H
Tanh_9Tanhadd_29*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_26Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_27Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_15/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Z
random_normal_15/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_15/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
§
%random_normal_15/RandomStandardNormalRandomStandardNormalrandom_normal_15/shape*
_output_shapes

:@2*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_15/mulMul%random_normal_15/RandomStandardNormalrandom_normal_15/stddev*
_output_shapes

:@2*
T0
m
random_normal_15Addrandom_normal_15/mulrandom_normal_15/mean*
T0*
_output_shapes

:@2

Variable_30
VariableV2*
_output_shapes

:@2*
	container *
shape
:@2*
dtype0*
shared_name 
≠
Variable_30/AssignAssignVariable_30random_normal_15*
_class
loc:@Variable_30*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
r
Variable_30/readIdentityVariable_30*
_class
loc:@Variable_30*
_output_shapes

:@2*
T0
]
zeros_15Const*
_output_shapes

:2*
dtype0*
valueB2*    
M
add_30/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_30Addzeros_15add_30/y*
_output_shapes

:2*
T0

Variable_31
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_31/AssignAssignVariable_31add_30*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_31*
T0*
use_locking(
r
Variable_31/readIdentityVariable_31*
T0*
_class
loc:@Variable_31*
_output_shapes

:2
Н
	MatMul_15MatMulPlaceholder_26Variable_30/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_31Add	MatMul_15Variable_31/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_13/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
I
Tanh_10Tanhadd_31*
T0*'
_output_shapes
:€€€€€€€€€2
Y
l1/outputs/tagConst*
valueB B
l1/outputs*
dtype0*
_output_shapes
: 
X

l1/outputsHistogramSummaryl1/outputs/tagTanh_10*
T0*
_output_shapes
: 
g
random_normal_16/shapeConst*
valueB"2   
   *
dtype0*
_output_shapes
:
Z
random_normal_16/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_16/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
§
%random_normal_16/RandomStandardNormalRandomStandardNormalrandom_normal_16/shape*
_output_shapes

:2
*
seed2 *
T0*

seed *
dtype0
Д
random_normal_16/mulMul%random_normal_16/RandomStandardNormalrandom_normal_16/stddev*
_output_shapes

:2
*
T0
m
random_normal_16Addrandom_normal_16/mulrandom_normal_16/mean*
T0*
_output_shapes

:2


Variable_32
VariableV2*
shape
:2
*
shared_name *
dtype0*
_output_shapes

:2
*
	container 
≠
Variable_32/AssignAssignVariable_32random_normal_16*
_class
loc:@Variable_32*
_output_shapes

:2
*
T0*
validate_shape(*
use_locking(
r
Variable_32/readIdentityVariable_32*
T0*
_output_shapes

:2
*
_class
loc:@Variable_32
]
zeros_16Const*
valueB
*    *
dtype0*
_output_shapes

:

M
add_32/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
J
add_32Addzeros_16add_32/y*
_output_shapes

:
*
T0

Variable_33
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
dtype0*
shared_name 
£
Variable_33/AssignAssignVariable_33add_32*
use_locking(*
T0*
_class
loc:@Variable_33*
validate_shape(*
_output_shapes

:

r
Variable_33/readIdentityVariable_33*
_class
loc:@Variable_33*
_output_shapes

:
*
T0
Ж
	MatMul_16MatMulTanh_10Variable_32/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_33Add	MatMul_16Variable_33/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_14/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
N
	Softmax_3Softmaxadd_33*
T0*'
_output_shapes
:€€€€€€€€€

Y
l2/outputs/tagConst*
_output_shapes
: *
dtype0*
valueB B
l2/outputs
Z

l2/outputsHistogramSummaryl2/outputs/tag	Softmax_3*
_output_shapes
: *
T0
I
Log_3Log	Softmax_3*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_3MulPlaceholder_27Log_3*
T0*'
_output_shapes
:€€€€€€€€€

a
Sum_3/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
w
Sum_3Summul_3Sum_3/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_3NegSum_3*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_3MeanNeg_3Const_3*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_2/tagsConst*
valueB Bloss_2*
_output_shapes
: *
dtype0
M
loss_2ScalarSummaryloss_2/tagsMean_3*
_output_shapes
: *
T0
T
gradients_2/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
V
gradients_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
_
gradients_2/FillFillgradients_2/Shapegradients_2/Const*
_output_shapes
: *
T0
o
%gradients_2/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_2/Mean_3_grad/ReshapeReshapegradients_2/Fill%gradients_2/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_2/Mean_3_grad/ShapeShapeNeg_3*
out_type0*
_output_shapes
:*
T0
§
gradients_2/Mean_3_grad/TileTilegradients_2/Mean_3_grad/Reshapegradients_2/Mean_3_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_2/Mean_3_grad/Shape_1ShapeNeg_3*
T0*
_output_shapes
:*
out_type0
b
gradients_2/Mean_3_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_2/Mean_3_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ґ
gradients_2/Mean_3_grad/ProdProdgradients_2/Mean_3_grad/Shape_1gradients_2/Mean_3_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_2/Mean_3_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¶
gradients_2/Mean_3_grad/Prod_1Prodgradients_2/Mean_3_grad/Shape_2gradients_2/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_2/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
gradients_2/Mean_3_grad/MaximumMaximumgradients_2/Mean_3_grad/Prod_1!gradients_2/Mean_3_grad/Maximum/y*
_output_shapes
: *
T0
М
 gradients_2/Mean_3_grad/floordivFloorDivgradients_2/Mean_3_grad/Prodgradients_2/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_2/Mean_3_grad/CastCast gradients_2/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_2/Mean_3_grad/truedivRealDivgradients_2/Mean_3_grad/Tilegradients_2/Mean_3_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_2/Neg_3_grad/NegNeggradients_2/Mean_3_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
a
gradients_2/Sum_3_grad/ShapeShapemul_3*
T0*
_output_shapes
:*
out_type0
]
gradients_2/Sum_3_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
gradients_2/Sum_3_grad/addAddSum_3/reduction_indicesgradients_2/Sum_3_grad/Size*
T0*
_output_shapes
:
Д
gradients_2/Sum_3_grad/modFloorModgradients_2/Sum_3_grad/addgradients_2/Sum_3_grad/Size*
T0*
_output_shapes
:
h
gradients_2/Sum_3_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
d
"gradients_2/Sum_3_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"gradients_2/Sum_3_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
≤
gradients_2/Sum_3_grad/rangeRange"gradients_2/Sum_3_grad/range/startgradients_2/Sum_3_grad/Size"gradients_2/Sum_3_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_2/Sum_3_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Л
gradients_2/Sum_3_grad/FillFillgradients_2/Sum_3_grad/Shape_1!gradients_2/Sum_3_grad/Fill/value*
T0*
_output_shapes
:
б
$gradients_2/Sum_3_grad/DynamicStitchDynamicStitchgradients_2/Sum_3_grad/rangegradients_2/Sum_3_grad/modgradients_2/Sum_3_grad/Shapegradients_2/Sum_3_grad/Fill*
T0*
N*#
_output_shapes
:€€€€€€€€€
b
 gradients_2/Sum_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
Я
gradients_2/Sum_3_grad/MaximumMaximum$gradients_2/Sum_3_grad/DynamicStitch gradients_2/Sum_3_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_2/Sum_3_grad/floordivFloorDivgradients_2/Sum_3_grad/Shapegradients_2/Sum_3_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_2/Sum_3_grad/ReshapeReshapegradients_2/Neg_3_grad/Neg$gradients_2/Sum_3_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
®
gradients_2/Sum_3_grad/TileTilegradients_2/Sum_3_grad/Reshapegradients_2/Sum_3_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_2/mul_3_grad/ShapeShapePlaceholder_27*
_output_shapes
:*
out_type0*
T0
c
gradients_2/mul_3_grad/Shape_1ShapeLog_3*
_output_shapes
:*
out_type0*
T0
ј
,gradients_2/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/mul_3_grad/Shapegradients_2/mul_3_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_2/mul_3_grad/mulMulgradients_2/Sum_3_grad/TileLog_3*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients_2/mul_3_grad/SumSumgradients_2/mul_3_grad/mul,gradients_2/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients_2/mul_3_grad/ReshapeReshapegradients_2/mul_3_grad/Sumgradients_2/mul_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

В
gradients_2/mul_3_grad/mul_1MulPlaceholder_27gradients_2/Sum_3_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_2/mul_3_grad/Sum_1Sumgradients_2/mul_3_grad/mul_1.gradients_2/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_2/mul_3_grad/Reshape_1Reshapegradients_2/mul_3_grad/Sum_1gradients_2/mul_3_grad/Shape_1*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
s
'gradients_2/mul_3_grad/tuple/group_depsNoOp^gradients_2/mul_3_grad/Reshape!^gradients_2/mul_3_grad/Reshape_1
к
/gradients_2/mul_3_grad/tuple/control_dependencyIdentitygradients_2/mul_3_grad/Reshape(^gradients_2/mul_3_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_2/mul_3_grad/Reshape
р
1gradients_2/mul_3_grad/tuple/control_dependency_1Identity gradients_2/mul_3_grad/Reshape_1(^gradients_2/mul_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_2/mul_3_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
*
T0
†
!gradients_2/Log_3_grad/Reciprocal
Reciprocal	Softmax_32^gradients_2/mul_3_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_2/Log_3_grad/mulMul1gradients_2/mul_3_grad/tuple/control_dependency_1!gradients_2/Log_3_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

~
gradients_2/Softmax_3_grad/mulMulgradients_2/Log_3_grad/mul	Softmax_3*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_2/Softmax_3_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
¬
gradients_2/Softmax_3_grad/SumSumgradients_2/Softmax_3_grad/mul0gradients_2/Softmax_3_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_2/Softmax_3_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients_2/Softmax_3_grad/ReshapeReshapegradients_2/Softmax_3_grad/Sum(gradients_2/Softmax_3_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ч
gradients_2/Softmax_3_grad/subSubgradients_2/Log_3_grad/mul"gradients_2/Softmax_3_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_2/Softmax_3_grad/mul_1Mulgradients_2/Softmax_3_grad/sub	Softmax_3*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_2/add_33_grad/ShapeShape	MatMul_16*
T0*
_output_shapes
:*
out_type0
p
gradients_2/add_33_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
√
-gradients_2/add_33_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_33_grad/Shapegradients_2/add_33_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_2/add_33_grad/SumSum gradients_2/Softmax_3_grad/mul_1-gradients_2/add_33_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_2/add_33_grad/ReshapeReshapegradients_2/add_33_grad/Sumgradients_2/add_33_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
Ј
gradients_2/add_33_grad/Sum_1Sum gradients_2/Softmax_3_grad/mul_1/gradients_2/add_33_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_2/add_33_grad/Reshape_1Reshapegradients_2/add_33_grad/Sum_1gradients_2/add_33_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_2/add_33_grad/tuple/group_depsNoOp ^gradients_2/add_33_grad/Reshape"^gradients_2/add_33_grad/Reshape_1
о
0gradients_2/add_33_grad/tuple/control_dependencyIdentitygradients_2/add_33_grad/Reshape)^gradients_2/add_33_grad/tuple/group_deps*2
_class(
&$loc:@gradients_2/add_33_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_2/add_33_grad/tuple/control_dependency_1Identity!gradients_2/add_33_grad/Reshape_1)^gradients_2/add_33_grad/tuple/group_deps*4
_class*
(&loc:@gradients_2/add_33_grad/Reshape_1*
_output_shapes

:
*
T0
«
!gradients_2/MatMul_16_grad/MatMulMatMul0gradients_2/add_33_grad/tuple/control_dependencyVariable_32/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_2/MatMul_16_grad/MatMul_1MatMulTanh_100gradients_2/add_33_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_2/MatMul_16_grad/tuple/group_depsNoOp"^gradients_2/MatMul_16_grad/MatMul$^gradients_2/MatMul_16_grad/MatMul_1
ш
3gradients_2/MatMul_16_grad/tuple/control_dependencyIdentity!gradients_2/MatMul_16_grad/MatMul,^gradients_2/MatMul_16_grad/tuple/group_deps*4
_class*
(&loc:@gradients_2/MatMul_16_grad/MatMul*'
_output_shapes
:€€€€€€€€€2*
T0
х
5gradients_2/MatMul_16_grad/tuple/control_dependency_1Identity#gradients_2/MatMul_16_grad/MatMul_1,^gradients_2/MatMul_16_grad/tuple/group_deps*
T0*
_output_shapes

:2
*6
_class,
*(loc:@gradients_2/MatMul_16_grad/MatMul_1
Э
!gradients_2/Tanh_10_grad/TanhGradTanhGradTanh_103gradients_2/MatMul_16_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_2/add_31_grad/ShapeShape	MatMul_15*
T0*
_output_shapes
:*
out_type0
p
gradients_2/add_31_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   2   
√
-gradients_2/add_31_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_31_grad/Shapegradients_2/add_31_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
gradients_2/add_31_grad/SumSum!gradients_2/Tanh_10_grad/TanhGrad-gradients_2/add_31_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_2/add_31_grad/ReshapeReshapegradients_2/add_31_grad/Sumgradients_2/add_31_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€2*
Tshape0
Є
gradients_2/add_31_grad/Sum_1Sum!gradients_2/Tanh_10_grad/TanhGrad/gradients_2/add_31_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_2/add_31_grad/Reshape_1Reshapegradients_2/add_31_grad/Sum_1gradients_2/add_31_grad/Shape_1*
T0*
_output_shapes

:2*
Tshape0
v
(gradients_2/add_31_grad/tuple/group_depsNoOp ^gradients_2/add_31_grad/Reshape"^gradients_2/add_31_grad/Reshape_1
о
0gradients_2/add_31_grad/tuple/control_dependencyIdentitygradients_2/add_31_grad/Reshape)^gradients_2/add_31_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_2/add_31_grad/Reshape*
T0
л
2gradients_2/add_31_grad/tuple/control_dependency_1Identity!gradients_2/add_31_grad/Reshape_1)^gradients_2/add_31_grad/tuple/group_deps*
T0*
_output_shapes

:2*4
_class*
(&loc:@gradients_2/add_31_grad/Reshape_1
«
!gradients_2/MatMul_15_grad/MatMulMatMul0gradients_2/add_31_grad/tuple/control_dependencyVariable_30/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_2/MatMul_15_grad/MatMul_1MatMulPlaceholder_260gradients_2/add_31_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
}
+gradients_2/MatMul_15_grad/tuple/group_depsNoOp"^gradients_2/MatMul_15_grad/MatMul$^gradients_2/MatMul_15_grad/MatMul_1
ш
3gradients_2/MatMul_15_grad/tuple/control_dependencyIdentity!gradients_2/MatMul_15_grad/MatMul,^gradients_2/MatMul_15_grad/tuple/group_deps*4
_class*
(&loc:@gradients_2/MatMul_15_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
х
5gradients_2/MatMul_15_grad/tuple/control_dependency_1Identity#gradients_2/MatMul_15_grad/MatMul_1,^gradients_2/MatMul_15_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_2/MatMul_15_grad/MatMul_1*
T0
d
GradientDescent_2/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
Т
9GradientDescent_2/update_Variable_30/ApplyGradientDescentApplyGradientDescentVariable_30GradientDescent_2/learning_rate5gradients_2/MatMul_15_grad/tuple/control_dependency_1*
_class
loc:@Variable_30*
_output_shapes

:@2*
T0*
use_locking( 
П
9GradientDescent_2/update_Variable_31/ApplyGradientDescentApplyGradientDescentVariable_31GradientDescent_2/learning_rate2gradients_2/add_31_grad/tuple/control_dependency_1*
_output_shapes

:2*
_class
loc:@Variable_31*
T0*
use_locking( 
Т
9GradientDescent_2/update_Variable_32/ApplyGradientDescentApplyGradientDescentVariable_32GradientDescent_2/learning_rate5gradients_2/MatMul_16_grad/tuple/control_dependency_1*
_class
loc:@Variable_32*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_2/update_Variable_33/ApplyGradientDescentApplyGradientDescentVariable_33GradientDescent_2/learning_rate2gradients_2/add_33_grad/tuple/control_dependency_1*
_output_shapes

:
*
_class
loc:@Variable_33*
T0*
use_locking( 
Й
GradientDescent_2NoOp:^GradientDescent_2/update_Variable_30/ApplyGradientDescent:^GradientDescent_2/update_Variable_31/ApplyGradientDescent:^GradientDescent_2/update_Variable_32/ApplyGradientDescent:^GradientDescent_2/update_Variable_33/ApplyGradientDescent
Q
Placeholder_28Placeholder*
_output_shapes
:*
shape: *
dtype0
s
Merge_1/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2*
N*
_output_shapes
: 
`
Placeholder_29Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
`
Placeholder_30Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_17/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Z
random_normal_17/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_17/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
§
%random_normal_17/RandomStandardNormalRandomStandardNormalrandom_normal_17/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_17/mulMul%random_normal_17/RandomStandardNormalrandom_normal_17/stddev*
T0*
_output_shapes

:@2
m
random_normal_17Addrandom_normal_17/mulrandom_normal_17/mean*
_output_shapes

:@2*
T0

Variable_34
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_34/AssignAssignVariable_34random_normal_17*
_class
loc:@Variable_34*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
r
Variable_34/readIdentityVariable_34*
T0*
_output_shapes

:@2*
_class
loc:@Variable_34
]
zeros_17Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_34/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_34Addzeros_17add_34/y*
T0*
_output_shapes

:2

Variable_35
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_35/AssignAssignVariable_35add_34*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_35*
T0*
use_locking(
r
Variable_35/readIdentityVariable_35*
_output_shapes

:2*
_class
loc:@Variable_35*
T0
Н
	MatMul_17MatMulPlaceholder_29Variable_34/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_35Add	MatMul_17Variable_35/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_15/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
I
Tanh_11Tanhadd_35*
T0*'
_output_shapes
:€€€€€€€€€2
]
l1/outputs_1/tagConst*
dtype0*
_output_shapes
: *
valueB Bl1/outputs_1
\
l1/outputs_1HistogramSummaryl1/outputs_1/tagTanh_11*
_output_shapes
: *
T0
g
random_normal_18/shapeConst*
_output_shapes
:*
dtype0*
valueB"2   
   
Z
random_normal_18/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_18/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_18/RandomStandardNormalRandomStandardNormalrandom_normal_18/shape*
_output_shapes

:2
*
seed2 *
T0*

seed *
dtype0
Д
random_normal_18/mulMul%random_normal_18/RandomStandardNormalrandom_normal_18/stddev*
T0*
_output_shapes

:2

m
random_normal_18Addrandom_normal_18/mulrandom_normal_18/mean*
_output_shapes

:2
*
T0

Variable_36
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_36/AssignAssignVariable_36random_normal_18*
_output_shapes

:2
*
validate_shape(*
_class
loc:@Variable_36*
T0*
use_locking(
r
Variable_36/readIdentityVariable_36*
_class
loc:@Variable_36*
_output_shapes

:2
*
T0
]
zeros_18Const*
valueB
*    *
_output_shapes

:
*
dtype0
M
add_36/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_36Addzeros_18add_36/y*
T0*
_output_shapes

:


Variable_37
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
dtype0*
shared_name 
£
Variable_37/AssignAssignVariable_37add_36*
_output_shapes

:
*
validate_shape(*
_class
loc:@Variable_37*
T0*
use_locking(
r
Variable_37/readIdentityVariable_37*
_output_shapes

:
*
_class
loc:@Variable_37*
T0
Ж
	MatMul_18MatMulTanh_11Variable_36/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_37Add	MatMul_18Variable_37/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_16/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_4Softmaxadd_37*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_1/tagConst*
_output_shapes
: *
dtype0*
valueB Bl2/outputs_1
^
l2/outputs_1HistogramSummaryl2/outputs_1/tag	Softmax_4*
_output_shapes
: *
T0
I
Log_4Log	Softmax_4*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_4MulPlaceholder_30Log_4*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_4/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
w
Sum_4Summul_4Sum_4/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_4NegSum_4*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_4MeanNeg_4Const_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
loss_3/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_3
M
loss_3ScalarSummaryloss_3/tagsMean_4*
T0*
_output_shapes
: 
T
gradients_3/ShapeConst*
valueB *
_output_shapes
: *
dtype0
V
gradients_3/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_3/FillFillgradients_3/Shapegradients_3/Const*
_output_shapes
: *
T0
o
%gradients_3/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_3/Mean_4_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_4_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients_3/Mean_4_grad/ShapeShapeNeg_4*
T0*
_output_shapes
:*
out_type0
§
gradients_3/Mean_4_grad/TileTilegradients_3/Mean_4_grad/Reshapegradients_3/Mean_4_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_3/Mean_4_grad/Shape_1ShapeNeg_4*
T0*
out_type0*
_output_shapes
:
b
gradients_3/Mean_4_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_3/Mean_4_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_3/Mean_4_grad/ProdProdgradients_3/Mean_4_grad/Shape_1gradients_3/Mean_4_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_3/Mean_4_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
¶
gradients_3/Mean_4_grad/Prod_1Prodgradients_3/Mean_4_grad/Shape_2gradients_3/Mean_4_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_3/Mean_4_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
О
gradients_3/Mean_4_grad/MaximumMaximumgradients_3/Mean_4_grad/Prod_1!gradients_3/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_3/Mean_4_grad/floordivFloorDivgradients_3/Mean_4_grad/Prodgradients_3/Mean_4_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_3/Mean_4_grad/CastCast gradients_3/Mean_4_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_3/Mean_4_grad/truedivRealDivgradients_3/Mean_4_grad/Tilegradients_3/Mean_4_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
p
gradients_3/Neg_4_grad/NegNeggradients_3/Mean_4_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
a
gradients_3/Sum_4_grad/ShapeShapemul_4*
_output_shapes
:*
out_type0*
T0
]
gradients_3/Sum_4_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
gradients_3/Sum_4_grad/addAddSum_4/reduction_indicesgradients_3/Sum_4_grad/Size*
_output_shapes
:*
T0
Д
gradients_3/Sum_4_grad/modFloorModgradients_3/Sum_4_grad/addgradients_3/Sum_4_grad/Size*
T0*
_output_shapes
:
h
gradients_3/Sum_4_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
d
"gradients_3/Sum_4_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"gradients_3/Sum_4_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
≤
gradients_3/Sum_4_grad/rangeRange"gradients_3/Sum_4_grad/range/startgradients_3/Sum_4_grad/Size"gradients_3/Sum_4_grad/range/delta*

Tidx0*
_output_shapes
:
c
!gradients_3/Sum_4_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Л
gradients_3/Sum_4_grad/FillFillgradients_3/Sum_4_grad/Shape_1!gradients_3/Sum_4_grad/Fill/value*
T0*
_output_shapes
:
б
$gradients_3/Sum_4_grad/DynamicStitchDynamicStitchgradients_3/Sum_4_grad/rangegradients_3/Sum_4_grad/modgradients_3/Sum_4_grad/Shapegradients_3/Sum_4_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
b
 gradients_3/Sum_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_3/Sum_4_grad/MaximumMaximum$gradients_3/Sum_4_grad/DynamicStitch gradients_3/Sum_4_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
О
gradients_3/Sum_4_grad/floordivFloorDivgradients_3/Sum_4_grad/Shapegradients_3/Sum_4_grad/Maximum*
T0*
_output_shapes
:
Ь
gradients_3/Sum_4_grad/ReshapeReshapegradients_3/Neg_4_grad/Neg$gradients_3/Sum_4_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
®
gradients_3/Sum_4_grad/TileTilegradients_3/Sum_4_grad/Reshapegradients_3/Sum_4_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_3/mul_4_grad/ShapeShapePlaceholder_30*
T0*
_output_shapes
:*
out_type0
c
gradients_3/mul_4_grad/Shape_1ShapeLog_4*
T0*
_output_shapes
:*
out_type0
ј
,gradients_3/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_4_grad/Shapegradients_3/mul_4_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_3/mul_4_grad/mulMulgradients_3/Sum_4_grad/TileLog_4*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_3/mul_4_grad/SumSumgradients_3/mul_4_grad/mul,gradients_3/mul_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_3/mul_4_grad/ReshapeReshapegradients_3/mul_4_grad/Sumgradients_3/mul_4_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
В
gradients_3/mul_4_grad/mul_1MulPlaceholder_30gradients_3/Sum_4_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_3/mul_4_grad/Sum_1Sumgradients_3/mul_4_grad/mul_1.gradients_3/mul_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_3/mul_4_grad/Reshape_1Reshapegradients_3/mul_4_grad/Sum_1gradients_3/mul_4_grad/Shape_1*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
s
'gradients_3/mul_4_grad/tuple/group_depsNoOp^gradients_3/mul_4_grad/Reshape!^gradients_3/mul_4_grad/Reshape_1
к
/gradients_3/mul_4_grad/tuple/control_dependencyIdentitygradients_3/mul_4_grad/Reshape(^gradients_3/mul_4_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_3/mul_4_grad/Reshape*
T0
р
1gradients_3/mul_4_grad/tuple/control_dependency_1Identity gradients_3/mul_4_grad/Reshape_1(^gradients_3/mul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_3/mul_4_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_3/Log_4_grad/Reciprocal
Reciprocal	Softmax_42^gradients_3/mul_4_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_3/Log_4_grad/mulMul1gradients_3/mul_4_grad/tuple/control_dependency_1!gradients_3/Log_4_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

~
gradients_3/Softmax_4_grad/mulMulgradients_3/Log_4_grad/mul	Softmax_4*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_3/Softmax_4_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_3/Softmax_4_grad/SumSumgradients_3/Softmax_4_grad/mul0gradients_3/Softmax_4_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_3/Softmax_4_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
Ј
"gradients_3/Softmax_4_grad/ReshapeReshapegradients_3/Softmax_4_grad/Sum(gradients_3/Softmax_4_grad/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Ч
gradients_3/Softmax_4_grad/subSubgradients_3/Log_4_grad/mul"gradients_3/Softmax_4_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_3/Softmax_4_grad/mul_1Mulgradients_3/Softmax_4_grad/sub	Softmax_4*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_3/add_37_grad/ShapeShape	MatMul_18*
T0*
out_type0*
_output_shapes
:
p
gradients_3/add_37_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
√
-gradients_3/add_37_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_37_grad/Shapegradients_3/add_37_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_3/add_37_grad/SumSum gradients_3/Softmax_4_grad/mul_1-gradients_3/add_37_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_3/add_37_grad/ReshapeReshapegradients_3/add_37_grad/Sumgradients_3/add_37_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

Ј
gradients_3/add_37_grad/Sum_1Sum gradients_3/Softmax_4_grad/mul_1/gradients_3/add_37_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_3/add_37_grad/Reshape_1Reshapegradients_3/add_37_grad/Sum_1gradients_3/add_37_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_3/add_37_grad/tuple/group_depsNoOp ^gradients_3/add_37_grad/Reshape"^gradients_3/add_37_grad/Reshape_1
о
0gradients_3/add_37_grad/tuple/control_dependencyIdentitygradients_3/add_37_grad/Reshape)^gradients_3/add_37_grad/tuple/group_deps*2
_class(
&$loc:@gradients_3/add_37_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_3/add_37_grad/tuple/control_dependency_1Identity!gradients_3/add_37_grad/Reshape_1)^gradients_3/add_37_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_3/add_37_grad/Reshape_1*
_output_shapes

:

«
!gradients_3/MatMul_18_grad/MatMulMatMul0gradients_3/add_37_grad/tuple/control_dependencyVariable_36/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Ј
#gradients_3/MatMul_18_grad/MatMul_1MatMulTanh_110gradients_3/add_37_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_3/MatMul_18_grad/tuple/group_depsNoOp"^gradients_3/MatMul_18_grad/MatMul$^gradients_3/MatMul_18_grad/MatMul_1
ш
3gradients_3/MatMul_18_grad/tuple/control_dependencyIdentity!gradients_3/MatMul_18_grad/MatMul,^gradients_3/MatMul_18_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*4
_class*
(&loc:@gradients_3/MatMul_18_grad/MatMul*
T0
х
5gradients_3/MatMul_18_grad/tuple/control_dependency_1Identity#gradients_3/MatMul_18_grad/MatMul_1,^gradients_3/MatMul_18_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_3/MatMul_18_grad/MatMul_1*
_output_shapes

:2

Э
!gradients_3/Tanh_11_grad/TanhGradTanhGradTanh_113gradients_3/MatMul_18_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_3/add_35_grad/ShapeShape	MatMul_17*
out_type0*
_output_shapes
:*
T0
p
gradients_3/add_35_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   2   
√
-gradients_3/add_35_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_35_grad/Shapegradients_3/add_35_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
gradients_3/add_35_grad/SumSum!gradients_3/Tanh_11_grad/TanhGrad-gradients_3/add_35_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_3/add_35_grad/ReshapeReshapegradients_3/add_35_grad/Sumgradients_3/add_35_grad/Shape*'
_output_shapes
:€€€€€€€€€2*
Tshape0*
T0
Є
gradients_3/add_35_grad/Sum_1Sum!gradients_3/Tanh_11_grad/TanhGrad/gradients_3/add_35_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_3/add_35_grad/Reshape_1Reshapegradients_3/add_35_grad/Sum_1gradients_3/add_35_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:2
v
(gradients_3/add_35_grad/tuple/group_depsNoOp ^gradients_3/add_35_grad/Reshape"^gradients_3/add_35_grad/Reshape_1
о
0gradients_3/add_35_grad/tuple/control_dependencyIdentitygradients_3/add_35_grad/Reshape)^gradients_3/add_35_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_3/add_35_grad/Reshape
л
2gradients_3/add_35_grad/tuple/control_dependency_1Identity!gradients_3/add_35_grad/Reshape_1)^gradients_3/add_35_grad/tuple/group_deps*
T0*
_output_shapes

:2*4
_class*
(&loc:@gradients_3/add_35_grad/Reshape_1
«
!gradients_3/MatMul_17_grad/MatMulMatMul0gradients_3/add_35_grad/tuple/control_dependencyVariable_34/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
T0
Њ
#gradients_3/MatMul_17_grad/MatMul_1MatMulPlaceholder_290gradients_3/add_35_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
}
+gradients_3/MatMul_17_grad/tuple/group_depsNoOp"^gradients_3/MatMul_17_grad/MatMul$^gradients_3/MatMul_17_grad/MatMul_1
ш
3gradients_3/MatMul_17_grad/tuple/control_dependencyIdentity!gradients_3/MatMul_17_grad/MatMul,^gradients_3/MatMul_17_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€@*4
_class*
(&loc:@gradients_3/MatMul_17_grad/MatMul
х
5gradients_3/MatMul_17_grad/tuple/control_dependency_1Identity#gradients_3/MatMul_17_grad/MatMul_1,^gradients_3/MatMul_17_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_3/MatMul_17_grad/MatMul_1*
_output_shapes

:@2
d
GradientDescent_3/learning_rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Т
9GradientDescent_3/update_Variable_34/ApplyGradientDescentApplyGradientDescentVariable_34GradientDescent_3/learning_rate5gradients_3/MatMul_17_grad/tuple/control_dependency_1*
_class
loc:@Variable_34*
_output_shapes

:@2*
T0*
use_locking( 
П
9GradientDescent_3/update_Variable_35/ApplyGradientDescentApplyGradientDescentVariable_35GradientDescent_3/learning_rate2gradients_3/add_35_grad/tuple/control_dependency_1*
_class
loc:@Variable_35*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_3/update_Variable_36/ApplyGradientDescentApplyGradientDescentVariable_36GradientDescent_3/learning_rate5gradients_3/MatMul_18_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:2
*
_class
loc:@Variable_36
П
9GradientDescent_3/update_Variable_37/ApplyGradientDescentApplyGradientDescentVariable_37GradientDescent_3/learning_rate2gradients_3/add_37_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_37*
_output_shapes

:

Й
GradientDescent_3NoOp:^GradientDescent_3/update_Variable_34/ApplyGradientDescent:^GradientDescent_3/update_Variable_35/ApplyGradientDescent:^GradientDescent_3/update_Variable_36/ApplyGradientDescent:^GradientDescent_3/update_Variable_37/ApplyGradientDescent
Q
Placeholder_31Placeholder*
shape: *
dtype0*
_output_shapes
:
Ч
Merge_2/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3*
N*
_output_shapes
: 
`
Placeholder_32Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_33Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

g
random_normal_19/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Z
random_normal_19/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_19/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_19/RandomStandardNormalRandomStandardNormalrandom_normal_19/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_19/mulMul%random_normal_19/RandomStandardNormalrandom_normal_19/stddev*
_output_shapes

:@2*
T0
m
random_normal_19Addrandom_normal_19/mulrandom_normal_19/mean*
_output_shapes

:@2*
T0

Variable_38
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_38/AssignAssignVariable_38random_normal_19*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_38
r
Variable_38/readIdentityVariable_38*
_output_shapes

:@2*
_class
loc:@Variable_38*
T0
]
zeros_19Const*
valueB2*    *
_output_shapes

:2*
dtype0
M
add_38/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
J
add_38Addzeros_19add_38/y*
T0*
_output_shapes

:2

Variable_39
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_39/AssignAssignVariable_39add_38*
use_locking(*
T0*
_class
loc:@Variable_39*
validate_shape(*
_output_shapes

:2
r
Variable_39/readIdentityVariable_39*
T0*
_class
loc:@Variable_39*
_output_shapes

:2
Н
	MatMul_19MatMulPlaceholder_32Variable_38/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_39Add	MatMul_19Variable_39/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_17/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
I
Tanh_12Tanhadd_39*'
_output_shapes
:€€€€€€€€€2*
T0
]
l1/outputs_2/tagConst*
valueB Bl1/outputs_2*
_output_shapes
: *
dtype0
\
l1/outputs_2HistogramSummaryl1/outputs_2/tagTanh_12*
_output_shapes
: *
T0
g
random_normal_20/shapeConst*
valueB"2   
   *
_output_shapes
:*
dtype0
Z
random_normal_20/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_20/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_20/RandomStandardNormalRandomStandardNormalrandom_normal_20/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_20/mulMul%random_normal_20/RandomStandardNormalrandom_normal_20/stddev*
T0*
_output_shapes

:2

m
random_normal_20Addrandom_normal_20/mulrandom_normal_20/mean*
T0*
_output_shapes

:2


Variable_40
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_40/AssignAssignVariable_40random_normal_20*
_class
loc:@Variable_40*
_output_shapes

:2
*
T0*
validate_shape(*
use_locking(
r
Variable_40/readIdentityVariable_40*
T0*
_class
loc:@Variable_40*
_output_shapes

:2

]
zeros_20Const*
valueB
*    *
_output_shapes

:
*
dtype0
M
add_40/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_40Addzeros_20add_40/y*
_output_shapes

:
*
T0

Variable_41
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
dtype0*
shared_name 
£
Variable_41/AssignAssignVariable_41add_40*
_output_shapes

:
*
validate_shape(*
_class
loc:@Variable_41*
T0*
use_locking(
r
Variable_41/readIdentityVariable_41*
_output_shapes

:
*
_class
loc:@Variable_41*
T0
Ж
	MatMul_20MatMulTanh_12Variable_40/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_41Add	MatMul_20Variable_41/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_18/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_5Softmaxadd_41*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_2/tagConst*
dtype0*
_output_shapes
: *
valueB Bl2/outputs_2
^
l2/outputs_2HistogramSummaryl2/outputs_2/tag	Softmax_5*
_output_shapes
: *
T0
I
Log_5Log	Softmax_5*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_5MulPlaceholder_33Log_5*
T0*'
_output_shapes
:€€€€€€€€€

a
Sum_5/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
w
Sum_5Summul_5Sum_5/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_5NegSum_5*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_5Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_5MeanNeg_5Const_5*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_4/tagsConst*
valueB Bloss_4*
dtype0*
_output_shapes
: 
M
loss_4ScalarSummaryloss_4/tagsMean_5*
_output_shapes
: *
T0
T
gradients_4/ShapeConst*
valueB *
_output_shapes
: *
dtype0
V
gradients_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
_
gradients_4/FillFillgradients_4/Shapegradients_4/Const*
T0*
_output_shapes
: 
o
%gradients_4/Mean_5_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ц
gradients_4/Mean_5_grad/ReshapeReshapegradients_4/Fill%gradients_4/Mean_5_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients_4/Mean_5_grad/ShapeShapeNeg_5*
out_type0*
_output_shapes
:*
T0
§
gradients_4/Mean_5_grad/TileTilegradients_4/Mean_5_grad/Reshapegradients_4/Mean_5_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_4/Mean_5_grad/Shape_1ShapeNeg_5*
_output_shapes
:*
out_type0*
T0
b
gradients_4/Mean_5_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_4/Mean_5_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ґ
gradients_4/Mean_5_grad/ProdProdgradients_4/Mean_5_grad/Shape_1gradients_4/Mean_5_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_4/Mean_5_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¶
gradients_4/Mean_5_grad/Prod_1Prodgradients_4/Mean_5_grad/Shape_2gradients_4/Mean_5_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_4/Mean_5_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
О
gradients_4/Mean_5_grad/MaximumMaximumgradients_4/Mean_5_grad/Prod_1!gradients_4/Mean_5_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_4/Mean_5_grad/floordivFloorDivgradients_4/Mean_5_grad/Prodgradients_4/Mean_5_grad/Maximum*
_output_shapes
: *
T0
v
gradients_4/Mean_5_grad/CastCast gradients_4/Mean_5_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_4/Mean_5_grad/truedivRealDivgradients_4/Mean_5_grad/Tilegradients_4/Mean_5_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
p
gradients_4/Neg_5_grad/NegNeggradients_4/Mean_5_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_4/Sum_5_grad/ShapeShapemul_5*
_output_shapes
:*
out_type0*
T0
]
gradients_4/Sum_5_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_4/Sum_5_grad/addAddSum_5/reduction_indicesgradients_4/Sum_5_grad/Size*
T0*
_output_shapes
:
Д
gradients_4/Sum_5_grad/modFloorModgradients_4/Sum_5_grad/addgradients_4/Sum_5_grad/Size*
T0*
_output_shapes
:
h
gradients_4/Sum_5_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
d
"gradients_4/Sum_5_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"gradients_4/Sum_5_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
≤
gradients_4/Sum_5_grad/rangeRange"gradients_4/Sum_5_grad/range/startgradients_4/Sum_5_grad/Size"gradients_4/Sum_5_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_4/Sum_5_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_4/Sum_5_grad/FillFillgradients_4/Sum_5_grad/Shape_1!gradients_4/Sum_5_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_4/Sum_5_grad/DynamicStitchDynamicStitchgradients_4/Sum_5_grad/rangegradients_4/Sum_5_grad/modgradients_4/Sum_5_grad/Shapegradients_4/Sum_5_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
b
 gradients_4/Sum_5_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
Я
gradients_4/Sum_5_grad/MaximumMaximum$gradients_4/Sum_5_grad/DynamicStitch gradients_4/Sum_5_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_4/Sum_5_grad/floordivFloorDivgradients_4/Sum_5_grad/Shapegradients_4/Sum_5_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_4/Sum_5_grad/ReshapeReshapegradients_4/Neg_5_grad/Neg$gradients_4/Sum_5_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0
®
gradients_4/Sum_5_grad/TileTilegradients_4/Sum_5_grad/Reshapegradients_4/Sum_5_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_4/mul_5_grad/ShapeShapePlaceholder_33*
_output_shapes
:*
out_type0*
T0
c
gradients_4/mul_5_grad/Shape_1ShapeLog_5*
_output_shapes
:*
out_type0*
T0
ј
,gradients_4/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/mul_5_grad/Shapegradients_4/mul_5_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_4/mul_5_grad/mulMulgradients_4/Sum_5_grad/TileLog_5*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_4/mul_5_grad/SumSumgradients_4/mul_5_grad/mul,gradients_4/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_4/mul_5_grad/ReshapeReshapegradients_4/mul_5_grad/Sumgradients_4/mul_5_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

В
gradients_4/mul_5_grad/mul_1MulPlaceholder_33gradients_4/Sum_5_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_4/mul_5_grad/Sum_1Sumgradients_4/mul_5_grad/mul_1.gradients_4/mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_4/mul_5_grad/Reshape_1Reshapegradients_4/mul_5_grad/Sum_1gradients_4/mul_5_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
s
'gradients_4/mul_5_grad/tuple/group_depsNoOp^gradients_4/mul_5_grad/Reshape!^gradients_4/mul_5_grad/Reshape_1
к
/gradients_4/mul_5_grad/tuple/control_dependencyIdentitygradients_4/mul_5_grad/Reshape(^gradients_4/mul_5_grad/tuple/group_deps*1
_class'
%#loc:@gradients_4/mul_5_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
р
1gradients_4/mul_5_grad/tuple/control_dependency_1Identity gradients_4/mul_5_grad/Reshape_1(^gradients_4/mul_5_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*3
_class)
'%loc:@gradients_4/mul_5_grad/Reshape_1
†
!gradients_4/Log_5_grad/Reciprocal
Reciprocal	Softmax_52^gradients_4/mul_5_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_4/Log_5_grad/mulMul1gradients_4/mul_5_grad/tuple/control_dependency_1!gradients_4/Log_5_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_4/Softmax_5_grad/mulMulgradients_4/Log_5_grad/mul	Softmax_5*'
_output_shapes
:€€€€€€€€€
*
T0
z
0gradients_4/Softmax_5_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_4/Softmax_5_grad/SumSumgradients_4/Softmax_5_grad/mul0gradients_4/Softmax_5_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_4/Softmax_5_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
Ј
"gradients_4/Softmax_5_grad/ReshapeReshapegradients_4/Softmax_5_grad/Sum(gradients_4/Softmax_5_grad/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
Ч
gradients_4/Softmax_5_grad/subSubgradients_4/Log_5_grad/mul"gradients_4/Softmax_5_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_4/Softmax_5_grad/mul_1Mulgradients_4/Softmax_5_grad/sub	Softmax_5*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_4/add_41_grad/ShapeShape	MatMul_20*
out_type0*
_output_shapes
:*
T0
p
gradients_4/add_41_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
√
-gradients_4/add_41_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_41_grad/Shapegradients_4/add_41_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_4/add_41_grad/SumSum gradients_4/Softmax_5_grad/mul_1-gradients_4/add_41_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_4/add_41_grad/ReshapeReshapegradients_4/add_41_grad/Sumgradients_4/add_41_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
Ј
gradients_4/add_41_grad/Sum_1Sum gradients_4/Softmax_5_grad/mul_1/gradients_4/add_41_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_4/add_41_grad/Reshape_1Reshapegradients_4/add_41_grad/Sum_1gradients_4/add_41_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_4/add_41_grad/tuple/group_depsNoOp ^gradients_4/add_41_grad/Reshape"^gradients_4/add_41_grad/Reshape_1
о
0gradients_4/add_41_grad/tuple/control_dependencyIdentitygradients_4/add_41_grad/Reshape)^gradients_4/add_41_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*2
_class(
&$loc:@gradients_4/add_41_grad/Reshape
л
2gradients_4/add_41_grad/tuple/control_dependency_1Identity!gradients_4/add_41_grad/Reshape_1)^gradients_4/add_41_grad/tuple/group_deps*4
_class*
(&loc:@gradients_4/add_41_grad/Reshape_1*
_output_shapes

:
*
T0
«
!gradients_4/MatMul_20_grad/MatMulMatMul0gradients_4/add_41_grad/tuple/control_dependencyVariable_40/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Ј
#gradients_4/MatMul_20_grad/MatMul_1MatMulTanh_120gradients_4/add_41_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
}
+gradients_4/MatMul_20_grad/tuple/group_depsNoOp"^gradients_4/MatMul_20_grad/MatMul$^gradients_4/MatMul_20_grad/MatMul_1
ш
3gradients_4/MatMul_20_grad/tuple/control_dependencyIdentity!gradients_4/MatMul_20_grad/MatMul,^gradients_4/MatMul_20_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_4/MatMul_20_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
х
5gradients_4/MatMul_20_grad/tuple/control_dependency_1Identity#gradients_4/MatMul_20_grad/MatMul_1,^gradients_4/MatMul_20_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_4/MatMul_20_grad/MatMul_1*
_output_shapes

:2

Э
!gradients_4/Tanh_12_grad/TanhGradTanhGradTanh_123gradients_4/MatMul_20_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_4/add_39_grad/ShapeShape	MatMul_19*
T0*
_output_shapes
:*
out_type0
p
gradients_4/add_39_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   2   
√
-gradients_4/add_39_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_39_grad/Shapegradients_4/add_39_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
gradients_4/add_39_grad/SumSum!gradients_4/Tanh_12_grad/TanhGrad-gradients_4/add_39_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_4/add_39_grad/ReshapeReshapegradients_4/add_39_grad/Sumgradients_4/add_39_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€2
Є
gradients_4/add_39_grad/Sum_1Sum!gradients_4/Tanh_12_grad/TanhGrad/gradients_4/add_39_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_4/add_39_grad/Reshape_1Reshapegradients_4/add_39_grad/Sum_1gradients_4/add_39_grad/Shape_1*
Tshape0*
_output_shapes

:2*
T0
v
(gradients_4/add_39_grad/tuple/group_depsNoOp ^gradients_4/add_39_grad/Reshape"^gradients_4/add_39_grad/Reshape_1
о
0gradients_4/add_39_grad/tuple/control_dependencyIdentitygradients_4/add_39_grad/Reshape)^gradients_4/add_39_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_4/add_39_grad/Reshape
л
2gradients_4/add_39_grad/tuple/control_dependency_1Identity!gradients_4/add_39_grad/Reshape_1)^gradients_4/add_39_grad/tuple/group_deps*4
_class*
(&loc:@gradients_4/add_39_grad/Reshape_1*
_output_shapes

:2*
T0
«
!gradients_4/MatMul_19_grad/MatMulMatMul0gradients_4/add_39_grad/tuple/control_dependencyVariable_38/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_4/MatMul_19_grad/MatMul_1MatMulPlaceholder_320gradients_4/add_39_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
}
+gradients_4/MatMul_19_grad/tuple/group_depsNoOp"^gradients_4/MatMul_19_grad/MatMul$^gradients_4/MatMul_19_grad/MatMul_1
ш
3gradients_4/MatMul_19_grad/tuple/control_dependencyIdentity!gradients_4/MatMul_19_grad/MatMul,^gradients_4/MatMul_19_grad/tuple/group_deps*4
_class*
(&loc:@gradients_4/MatMul_19_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
х
5gradients_4/MatMul_19_grad/tuple/control_dependency_1Identity#gradients_4/MatMul_19_grad/MatMul_1,^gradients_4/MatMul_19_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_4/MatMul_19_grad/MatMul_1*
T0
d
GradientDescent_4/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
Т
9GradientDescent_4/update_Variable_38/ApplyGradientDescentApplyGradientDescentVariable_38GradientDescent_4/learning_rate5gradients_4/MatMul_19_grad/tuple/control_dependency_1*
_output_shapes

:@2*
_class
loc:@Variable_38*
T0*
use_locking( 
П
9GradientDescent_4/update_Variable_39/ApplyGradientDescentApplyGradientDescentVariable_39GradientDescent_4/learning_rate2gradients_4/add_39_grad/tuple/control_dependency_1*
_class
loc:@Variable_39*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_4/update_Variable_40/ApplyGradientDescentApplyGradientDescentVariable_40GradientDescent_4/learning_rate5gradients_4/MatMul_20_grad/tuple/control_dependency_1*
_class
loc:@Variable_40*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_4/update_Variable_41/ApplyGradientDescentApplyGradientDescentVariable_41GradientDescent_4/learning_rate2gradients_4/add_41_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:
*
_class
loc:@Variable_41
Й
GradientDescent_4NoOp:^GradientDescent_4/update_Variable_38/ApplyGradientDescent:^GradientDescent_4/update_Variable_39/ApplyGradientDescent:^GradientDescent_4/update_Variable_40/ApplyGradientDescent:^GradientDescent_4/update_Variable_41/ApplyGradientDescent
Q
Placeholder_34Placeholder*
_output_shapes
:*
dtype0*
shape: 
ї
Merge_3/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4*
_output_shapes
: *
N
т
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign
`
Placeholder_35Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_36Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

g
random_normal_21/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Z
random_normal_21/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_21/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_21/RandomStandardNormalRandomStandardNormalrandom_normal_21/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_21/mulMul%random_normal_21/RandomStandardNormalrandom_normal_21/stddev*
T0*
_output_shapes

:@2
m
random_normal_21Addrandom_normal_21/mulrandom_normal_21/mean*
T0*
_output_shapes

:@2

Variable_42
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_42/AssignAssignVariable_42random_normal_21*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_42*
T0*
use_locking(
r
Variable_42/readIdentityVariable_42*
T0*
_output_shapes

:@2*
_class
loc:@Variable_42
]
zeros_21Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_42/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_42Addzeros_21add_42/y*
_output_shapes

:2*
T0

Variable_43
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_43/AssignAssignVariable_43add_42*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_43*
T0*
use_locking(
r
Variable_43/readIdentityVariable_43*
_output_shapes

:2*
_class
loc:@Variable_43*
T0
Н
	MatMul_21MatMulPlaceholder_35Variable_42/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_43Add	MatMul_21Variable_43/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_19/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
I
Tanh_13Tanhadd_43*'
_output_shapes
:€€€€€€€€€2*
T0
]
l1/outputs_3/tagConst*
_output_shapes
: *
dtype0*
valueB Bl1/outputs_3
\
l1/outputs_3HistogramSummaryl1/outputs_3/tagTanh_13*
T0*
_output_shapes
: 
g
random_normal_22/shapeConst*
dtype0*
_output_shapes
:*
valueB"2   
   
Z
random_normal_22/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_22/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_22/RandomStandardNormalRandomStandardNormalrandom_normal_22/shape*

seed *
T0*
dtype0*
_output_shapes

:2
*
seed2 
Д
random_normal_22/mulMul%random_normal_22/RandomStandardNormalrandom_normal_22/stddev*
T0*
_output_shapes

:2

m
random_normal_22Addrandom_normal_22/mulrandom_normal_22/mean*
_output_shapes

:2
*
T0

Variable_44
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_44/AssignAssignVariable_44random_normal_22*
_class
loc:@Variable_44*
_output_shapes

:2
*
T0*
validate_shape(*
use_locking(
r
Variable_44/readIdentityVariable_44*
_class
loc:@Variable_44*
_output_shapes

:2
*
T0
]
zeros_22Const*
valueB
*    *
_output_shapes

:
*
dtype0
M
add_44/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_44Addzeros_22add_44/y*
T0*
_output_shapes

:


Variable_45
VariableV2*
_output_shapes

:
*
	container *
dtype0*
shared_name *
shape
:

£
Variable_45/AssignAssignVariable_45add_44*
use_locking(*
T0*
_class
loc:@Variable_45*
validate_shape(*
_output_shapes

:

r
Variable_45/readIdentityVariable_45*
_class
loc:@Variable_45*
_output_shapes

:
*
T0
Ж
	MatMul_22MatMulTanh_13Variable_44/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_45Add	MatMul_22Variable_45/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_20/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
N
	Softmax_6Softmaxadd_45*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_3/tagConst*
_output_shapes
: *
dtype0*
valueB Bl2/outputs_3
^
l2/outputs_3HistogramSummaryl2/outputs_3/tag	Softmax_6*
T0*
_output_shapes
: 
I
Log_6Log	Softmax_6*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_6MulPlaceholder_36Log_6*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_6/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
w
Sum_6Summul_6Sum_6/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_6NegSum_6*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_6Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_6MeanNeg_6Const_6*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_5/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_5
M
loss_5ScalarSummaryloss_5/tagsMean_6*
_output_shapes
: *
T0
T
gradients_5/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
V
gradients_5/ConstConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
_
gradients_5/FillFillgradients_5/Shapegradients_5/Const*
_output_shapes
: *
T0
o
%gradients_5/Mean_6_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Ц
gradients_5/Mean_6_grad/ReshapeReshapegradients_5/Fill%gradients_5/Mean_6_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients_5/Mean_6_grad/ShapeShapeNeg_6*
out_type0*
_output_shapes
:*
T0
§
gradients_5/Mean_6_grad/TileTilegradients_5/Mean_6_grad/Reshapegradients_5/Mean_6_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_5/Mean_6_grad/Shape_1ShapeNeg_6*
T0*
out_type0*
_output_shapes
:
b
gradients_5/Mean_6_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_5/Mean_6_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
gradients_5/Mean_6_grad/ProdProdgradients_5/Mean_6_grad/Shape_1gradients_5/Mean_6_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_5/Mean_6_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¶
gradients_5/Mean_6_grad/Prod_1Prodgradients_5/Mean_6_grad/Shape_2gradients_5/Mean_6_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_5/Mean_6_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
О
gradients_5/Mean_6_grad/MaximumMaximumgradients_5/Mean_6_grad/Prod_1!gradients_5/Mean_6_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_5/Mean_6_grad/floordivFloorDivgradients_5/Mean_6_grad/Prodgradients_5/Mean_6_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_5/Mean_6_grad/CastCast gradients_5/Mean_6_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_5/Mean_6_grad/truedivRealDivgradients_5/Mean_6_grad/Tilegradients_5/Mean_6_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_5/Neg_6_grad/NegNeggradients_5/Mean_6_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
a
gradients_5/Sum_6_grad/ShapeShapemul_6*
T0*
_output_shapes
:*
out_type0
]
gradients_5/Sum_6_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
gradients_5/Sum_6_grad/addAddSum_6/reduction_indicesgradients_5/Sum_6_grad/Size*
_output_shapes
:*
T0
Д
gradients_5/Sum_6_grad/modFloorModgradients_5/Sum_6_grad/addgradients_5/Sum_6_grad/Size*
T0*
_output_shapes
:
h
gradients_5/Sum_6_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
d
"gradients_5/Sum_6_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"gradients_5/Sum_6_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
≤
gradients_5/Sum_6_grad/rangeRange"gradients_5/Sum_6_grad/range/startgradients_5/Sum_6_grad/Size"gradients_5/Sum_6_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_5/Sum_6_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
Л
gradients_5/Sum_6_grad/FillFillgradients_5/Sum_6_grad/Shape_1!gradients_5/Sum_6_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_5/Sum_6_grad/DynamicStitchDynamicStitchgradients_5/Sum_6_grad/rangegradients_5/Sum_6_grad/modgradients_5/Sum_6_grad/Shapegradients_5/Sum_6_grad/Fill*#
_output_shapes
:€€€€€€€€€*
T0*
N
b
 gradients_5/Sum_6_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
Я
gradients_5/Sum_6_grad/MaximumMaximum$gradients_5/Sum_6_grad/DynamicStitch gradients_5/Sum_6_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_5/Sum_6_grad/floordivFloorDivgradients_5/Sum_6_grad/Shapegradients_5/Sum_6_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_5/Sum_6_grad/ReshapeReshapegradients_5/Neg_6_grad/Neg$gradients_5/Sum_6_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
®
gradients_5/Sum_6_grad/TileTilegradients_5/Sum_6_grad/Reshapegradients_5/Sum_6_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_5/mul_6_grad/ShapeShapePlaceholder_36*
T0*
out_type0*
_output_shapes
:
c
gradients_5/mul_6_grad/Shape_1ShapeLog_6*
T0*
out_type0*
_output_shapes
:
ј
,gradients_5/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/mul_6_grad/Shapegradients_5/mul_6_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_5/mul_6_grad/mulMulgradients_5/Sum_6_grad/TileLog_6*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_5/mul_6_grad/SumSumgradients_5/mul_6_grad/mul,gradients_5/mul_6_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_5/mul_6_grad/ReshapeReshapegradients_5/mul_6_grad/Sumgradients_5/mul_6_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
В
gradients_5/mul_6_grad/mul_1MulPlaceholder_36gradients_5/Sum_6_grad/Tile*'
_output_shapes
:€€€€€€€€€
*
T0
±
gradients_5/mul_6_grad/Sum_1Sumgradients_5/mul_6_grad/mul_1.gradients_5/mul_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
©
 gradients_5/mul_6_grad/Reshape_1Reshapegradients_5/mul_6_grad/Sum_1gradients_5/mul_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

s
'gradients_5/mul_6_grad/tuple/group_depsNoOp^gradients_5/mul_6_grad/Reshape!^gradients_5/mul_6_grad/Reshape_1
к
/gradients_5/mul_6_grad/tuple/control_dependencyIdentitygradients_5/mul_6_grad/Reshape(^gradients_5/mul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_5/mul_6_grad/Reshape*'
_output_shapes
:€€€€€€€€€

р
1gradients_5/mul_6_grad/tuple/control_dependency_1Identity gradients_5/mul_6_grad/Reshape_1(^gradients_5/mul_6_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*3
_class)
'%loc:@gradients_5/mul_6_grad/Reshape_1*
T0
†
!gradients_5/Log_6_grad/Reciprocal
Reciprocal	Softmax_62^gradients_5/mul_6_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_5/Log_6_grad/mulMul1gradients_5/mul_6_grad/tuple/control_dependency_1!gradients_5/Log_6_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_5/Softmax_6_grad/mulMulgradients_5/Log_6_grad/mul	Softmax_6*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_5/Softmax_6_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_5/Softmax_6_grad/SumSumgradients_5/Softmax_6_grad/mul0gradients_5/Softmax_6_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_5/Softmax_6_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients_5/Softmax_6_grad/ReshapeReshapegradients_5/Softmax_6_grad/Sum(gradients_5/Softmax_6_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
Ч
gradients_5/Softmax_6_grad/subSubgradients_5/Log_6_grad/mul"gradients_5/Softmax_6_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_5/Softmax_6_grad/mul_1Mulgradients_5/Softmax_6_grad/sub	Softmax_6*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_5/add_45_grad/ShapeShape	MatMul_22*
_output_shapes
:*
out_type0*
T0
p
gradients_5/add_45_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
√
-gradients_5/add_45_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_45_grad/Shapegradients_5/add_45_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_5/add_45_grad/SumSum gradients_5/Softmax_6_grad/mul_1-gradients_5/add_45_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_5/add_45_grad/ReshapeReshapegradients_5/add_45_grad/Sumgradients_5/add_45_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
Ј
gradients_5/add_45_grad/Sum_1Sum gradients_5/Softmax_6_grad/mul_1/gradients_5/add_45_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_5/add_45_grad/Reshape_1Reshapegradients_5/add_45_grad/Sum_1gradients_5/add_45_grad/Shape_1*
T0*
_output_shapes

:
*
Tshape0
v
(gradients_5/add_45_grad/tuple/group_depsNoOp ^gradients_5/add_45_grad/Reshape"^gradients_5/add_45_grad/Reshape_1
о
0gradients_5/add_45_grad/tuple/control_dependencyIdentitygradients_5/add_45_grad/Reshape)^gradients_5/add_45_grad/tuple/group_deps*2
_class(
&$loc:@gradients_5/add_45_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_5/add_45_grad/tuple/control_dependency_1Identity!gradients_5/add_45_grad/Reshape_1)^gradients_5/add_45_grad/tuple/group_deps*
T0*
_output_shapes

:
*4
_class*
(&loc:@gradients_5/add_45_grad/Reshape_1
«
!gradients_5/MatMul_22_grad/MatMulMatMul0gradients_5/add_45_grad/tuple/control_dependencyVariable_44/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_5/MatMul_22_grad/MatMul_1MatMulTanh_130gradients_5/add_45_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
}
+gradients_5/MatMul_22_grad/tuple/group_depsNoOp"^gradients_5/MatMul_22_grad/MatMul$^gradients_5/MatMul_22_grad/MatMul_1
ш
3gradients_5/MatMul_22_grad/tuple/control_dependencyIdentity!gradients_5/MatMul_22_grad/MatMul,^gradients_5/MatMul_22_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_5/MatMul_22_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
х
5gradients_5/MatMul_22_grad/tuple/control_dependency_1Identity#gradients_5/MatMul_22_grad/MatMul_1,^gradients_5/MatMul_22_grad/tuple/group_deps*
T0*
_output_shapes

:2
*6
_class,
*(loc:@gradients_5/MatMul_22_grad/MatMul_1
Э
!gradients_5/Tanh_13_grad/TanhGradTanhGradTanh_133gradients_5/MatMul_22_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
f
gradients_5/add_43_grad/ShapeShape	MatMul_21*
out_type0*
_output_shapes
:*
T0
p
gradients_5/add_43_grad/Shape_1Const*
valueB"   2   *
dtype0*
_output_shapes
:
√
-gradients_5/add_43_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_43_grad/Shapegradients_5/add_43_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
gradients_5/add_43_grad/SumSum!gradients_5/Tanh_13_grad/TanhGrad-gradients_5/add_43_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_5/add_43_grad/ReshapeReshapegradients_5/add_43_grad/Sumgradients_5/add_43_grad/Shape*'
_output_shapes
:€€€€€€€€€2*
Tshape0*
T0
Є
gradients_5/add_43_grad/Sum_1Sum!gradients_5/Tanh_13_grad/TanhGrad/gradients_5/add_43_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_5/add_43_grad/Reshape_1Reshapegradients_5/add_43_grad/Sum_1gradients_5/add_43_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:2
v
(gradients_5/add_43_grad/tuple/group_depsNoOp ^gradients_5/add_43_grad/Reshape"^gradients_5/add_43_grad/Reshape_1
о
0gradients_5/add_43_grad/tuple/control_dependencyIdentitygradients_5/add_43_grad/Reshape)^gradients_5/add_43_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_5/add_43_grad/Reshape*
T0
л
2gradients_5/add_43_grad/tuple/control_dependency_1Identity!gradients_5/add_43_grad/Reshape_1)^gradients_5/add_43_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_5/add_43_grad/Reshape_1*
_output_shapes

:2
«
!gradients_5/MatMul_21_grad/MatMulMatMul0gradients_5/add_43_grad/tuple/control_dependencyVariable_42/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_5/MatMul_21_grad/MatMul_1MatMulPlaceholder_350gradients_5/add_43_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_5/MatMul_21_grad/tuple/group_depsNoOp"^gradients_5/MatMul_21_grad/MatMul$^gradients_5/MatMul_21_grad/MatMul_1
ш
3gradients_5/MatMul_21_grad/tuple/control_dependencyIdentity!gradients_5/MatMul_21_grad/MatMul,^gradients_5/MatMul_21_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€@*4
_class*
(&loc:@gradients_5/MatMul_21_grad/MatMul
х
5gradients_5/MatMul_21_grad/tuple/control_dependency_1Identity#gradients_5/MatMul_21_grad/MatMul_1,^gradients_5/MatMul_21_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_5/MatMul_21_grad/MatMul_1*
T0
d
GradientDescent_5/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
Т
9GradientDescent_5/update_Variable_42/ApplyGradientDescentApplyGradientDescentVariable_42GradientDescent_5/learning_rate5gradients_5/MatMul_21_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:@2*
_class
loc:@Variable_42
П
9GradientDescent_5/update_Variable_43/ApplyGradientDescentApplyGradientDescentVariable_43GradientDescent_5/learning_rate2gradients_5/add_43_grad/tuple/control_dependency_1*
_class
loc:@Variable_43*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_5/update_Variable_44/ApplyGradientDescentApplyGradientDescentVariable_44GradientDescent_5/learning_rate5gradients_5/MatMul_22_grad/tuple/control_dependency_1*
_class
loc:@Variable_44*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_5/update_Variable_45/ApplyGradientDescentApplyGradientDescentVariable_45GradientDescent_5/learning_rate2gradients_5/add_45_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:
*
_class
loc:@Variable_45
Й
GradientDescent_5NoOp:^GradientDescent_5/update_Variable_42/ApplyGradientDescent:^GradientDescent_5/update_Variable_43/ApplyGradientDescent:^GradientDescent_5/update_Variable_44/ApplyGradientDescent:^GradientDescent_5/update_Variable_45/ApplyGradientDescent
Q
Placeholder_37Placeholder*
dtype0*
shape: *
_output_shapes
:
`
Placeholder_38Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_39Placeholder*'
_output_shapes
:€€€€€€€€€
*
shape: *
dtype0
я
Merge_4/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4l1/outputs_3l2/outputs_3loss_5*
_output_shapes
: *
N
»
init_1NoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign^Variable_44/Assign^Variable_45/Assign
Q
Placeholder_40Placeholder*
shape: *
dtype0*
_output_shapes
:
`
Placeholder_41Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_42Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

g
random_normal_23/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Z
random_normal_23/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_23/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_23/RandomStandardNormalRandomStandardNormalrandom_normal_23/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_23/mulMul%random_normal_23/RandomStandardNormalrandom_normal_23/stddev*
_output_shapes

:@2*
T0
m
random_normal_23Addrandom_normal_23/mulrandom_normal_23/mean*
_output_shapes

:@2*
T0

Variable_46
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_46/AssignAssignVariable_46random_normal_23*
use_locking(*
T0*
_class
loc:@Variable_46*
validate_shape(*
_output_shapes

:@2
r
Variable_46/readIdentityVariable_46*
_class
loc:@Variable_46*
_output_shapes

:@2*
T0
]
zeros_23Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_46/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_46Addzeros_23add_46/y*
_output_shapes

:2*
T0

Variable_47
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_47/AssignAssignVariable_47add_46*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_47
r
Variable_47/readIdentityVariable_47*
T0*
_output_shapes

:2*
_class
loc:@Variable_47
Н
	MatMul_23MatMulPlaceholder_41Variable_46/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_47Add	MatMul_23Variable_47/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_21/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
I
Tanh_14Tanhadd_47*
T0*'
_output_shapes
:€€€€€€€€€2
]
l1/outputs_4/tagConst*
_output_shapes
: *
dtype0*
valueB Bl1/outputs_4
\
l1/outputs_4HistogramSummaryl1/outputs_4/tagTanh_14*
_output_shapes
: *
T0
g
random_normal_24/shapeConst*
valueB"2   
   *
dtype0*
_output_shapes
:
Z
random_normal_24/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_24/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
§
%random_normal_24/RandomStandardNormalRandomStandardNormalrandom_normal_24/shape*
_output_shapes

:2
*
seed2 *
T0*

seed *
dtype0
Д
random_normal_24/mulMul%random_normal_24/RandomStandardNormalrandom_normal_24/stddev*
T0*
_output_shapes

:2

m
random_normal_24Addrandom_normal_24/mulrandom_normal_24/mean*
T0*
_output_shapes

:2


Variable_48
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_48/AssignAssignVariable_48random_normal_24*
_output_shapes

:2
*
validate_shape(*
_class
loc:@Variable_48*
T0*
use_locking(
r
Variable_48/readIdentityVariable_48*
T0*
_output_shapes

:2
*
_class
loc:@Variable_48
]
zeros_24Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_48/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_48Addzeros_24add_48/y*
T0*
_output_shapes

:


Variable_49
VariableV2*
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*
	container 
£
Variable_49/AssignAssignVariable_49add_48*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
*
_class
loc:@Variable_49
r
Variable_49/readIdentityVariable_49*
_class
loc:@Variable_49*
_output_shapes

:
*
T0
Ж
	MatMul_24MatMulTanh_14Variable_48/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
\
add_49Add	MatMul_24Variable_49/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_22/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
N
	Softmax_7Softmaxadd_49*'
_output_shapes
:€€€€€€€€€
*
T0
]
l2/outputs_4/tagConst*
dtype0*
_output_shapes
: *
valueB Bl2/outputs_4
^
l2/outputs_4HistogramSummaryl2/outputs_4/tag	Softmax_7*
_output_shapes
: *
T0
I
Log_7Log	Softmax_7*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_7MulPlaceholder_42Log_7*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_7/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
w
Sum_7Summul_7Sum_7/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_7NegSum_7*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_7Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_7MeanNeg_7Const_7*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_6/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_6
M
loss_6ScalarSummaryloss_6/tagsMean_7*
_output_shapes
: *
T0
T
gradients_6/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
V
gradients_6/ConstConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
_
gradients_6/FillFillgradients_6/Shapegradients_6/Const*
_output_shapes
: *
T0
o
%gradients_6/Mean_7_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_6/Mean_7_grad/ReshapeReshapegradients_6/Fill%gradients_6/Mean_7_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients_6/Mean_7_grad/ShapeShapeNeg_7*
T0*
_output_shapes
:*
out_type0
§
gradients_6/Mean_7_grad/TileTilegradients_6/Mean_7_grad/Reshapegradients_6/Mean_7_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_6/Mean_7_grad/Shape_1ShapeNeg_7*
T0*
_output_shapes
:*
out_type0
b
gradients_6/Mean_7_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_6/Mean_7_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ґ
gradients_6/Mean_7_grad/ProdProdgradients_6/Mean_7_grad/Shape_1gradients_6/Mean_7_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_6/Mean_7_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¶
gradients_6/Mean_7_grad/Prod_1Prodgradients_6/Mean_7_grad/Shape_2gradients_6/Mean_7_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_6/Mean_7_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_6/Mean_7_grad/MaximumMaximumgradients_6/Mean_7_grad/Prod_1!gradients_6/Mean_7_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_6/Mean_7_grad/floordivFloorDivgradients_6/Mean_7_grad/Prodgradients_6/Mean_7_grad/Maximum*
_output_shapes
: *
T0
v
gradients_6/Mean_7_grad/CastCast gradients_6/Mean_7_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_6/Mean_7_grad/truedivRealDivgradients_6/Mean_7_grad/Tilegradients_6/Mean_7_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_6/Neg_7_grad/NegNeggradients_6/Mean_7_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_6/Sum_7_grad/ShapeShapemul_7*
_output_shapes
:*
out_type0*
T0
]
gradients_6/Sum_7_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
gradients_6/Sum_7_grad/addAddSum_7/reduction_indicesgradients_6/Sum_7_grad/Size*
T0*
_output_shapes
:
Д
gradients_6/Sum_7_grad/modFloorModgradients_6/Sum_7_grad/addgradients_6/Sum_7_grad/Size*
T0*
_output_shapes
:
h
gradients_6/Sum_7_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
d
"gradients_6/Sum_7_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"gradients_6/Sum_7_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
≤
gradients_6/Sum_7_grad/rangeRange"gradients_6/Sum_7_grad/range/startgradients_6/Sum_7_grad/Size"gradients_6/Sum_7_grad/range/delta*

Tidx0*
_output_shapes
:
c
!gradients_6/Sum_7_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :
Л
gradients_6/Sum_7_grad/FillFillgradients_6/Sum_7_grad/Shape_1!gradients_6/Sum_7_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_6/Sum_7_grad/DynamicStitchDynamicStitchgradients_6/Sum_7_grad/rangegradients_6/Sum_7_grad/modgradients_6/Sum_7_grad/Shapegradients_6/Sum_7_grad/Fill*
N*
T0*#
_output_shapes
:€€€€€€€€€
b
 gradients_6/Sum_7_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
Я
gradients_6/Sum_7_grad/MaximumMaximum$gradients_6/Sum_7_grad/DynamicStitch gradients_6/Sum_7_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_6/Sum_7_grad/floordivFloorDivgradients_6/Sum_7_grad/Shapegradients_6/Sum_7_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_6/Sum_7_grad/ReshapeReshapegradients_6/Neg_7_grad/Neg$gradients_6/Sum_7_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
®
gradients_6/Sum_7_grad/TileTilegradients_6/Sum_7_grad/Reshapegradients_6/Sum_7_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_6/mul_7_grad/ShapeShapePlaceholder_42*
T0*
out_type0*
_output_shapes
:
c
gradients_6/mul_7_grad/Shape_1ShapeLog_7*
T0*
_output_shapes
:*
out_type0
ј
,gradients_6/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/mul_7_grad/Shapegradients_6/mul_7_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_6/mul_7_grad/mulMulgradients_6/Sum_7_grad/TileLog_7*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients_6/mul_7_grad/SumSumgradients_6/mul_7_grad/mul,gradients_6/mul_7_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients_6/mul_7_grad/ReshapeReshapegradients_6/mul_7_grad/Sumgradients_6/mul_7_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
В
gradients_6/mul_7_grad/mul_1MulPlaceholder_42gradients_6/Sum_7_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_6/mul_7_grad/Sum_1Sumgradients_6/mul_7_grad/mul_1.gradients_6/mul_7_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_6/mul_7_grad/Reshape_1Reshapegradients_6/mul_7_grad/Sum_1gradients_6/mul_7_grad/Shape_1*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
s
'gradients_6/mul_7_grad/tuple/group_depsNoOp^gradients_6/mul_7_grad/Reshape!^gradients_6/mul_7_grad/Reshape_1
к
/gradients_6/mul_7_grad/tuple/control_dependencyIdentitygradients_6/mul_7_grad/Reshape(^gradients_6/mul_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_6/mul_7_grad/Reshape*'
_output_shapes
:€€€€€€€€€

р
1gradients_6/mul_7_grad/tuple/control_dependency_1Identity gradients_6/mul_7_grad/Reshape_1(^gradients_6/mul_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_6/mul_7_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_6/Log_7_grad/Reciprocal
Reciprocal	Softmax_72^gradients_6/mul_7_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_6/Log_7_grad/mulMul1gradients_6/mul_7_grad/tuple/control_dependency_1!gradients_6/Log_7_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_6/Softmax_7_grad/mulMulgradients_6/Log_7_grad/mul	Softmax_7*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_6/Softmax_7_grad/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
¬
gradients_6/Softmax_7_grad/SumSumgradients_6/Softmax_7_grad/mul0gradients_6/Softmax_7_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_6/Softmax_7_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
Ј
"gradients_6/Softmax_7_grad/ReshapeReshapegradients_6/Softmax_7_grad/Sum(gradients_6/Softmax_7_grad/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
Ч
gradients_6/Softmax_7_grad/subSubgradients_6/Log_7_grad/mul"gradients_6/Softmax_7_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_6/Softmax_7_grad/mul_1Mulgradients_6/Softmax_7_grad/sub	Softmax_7*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_6/add_49_grad/ShapeShape	MatMul_24*
T0*
out_type0*
_output_shapes
:
p
gradients_6/add_49_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   
√
-gradients_6/add_49_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/add_49_grad/Shapegradients_6/add_49_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_6/add_49_grad/SumSum gradients_6/Softmax_7_grad/mul_1-gradients_6/add_49_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_6/add_49_grad/ReshapeReshapegradients_6/add_49_grad/Sumgradients_6/add_49_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
Ј
gradients_6/add_49_grad/Sum_1Sum gradients_6/Softmax_7_grad/mul_1/gradients_6/add_49_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_6/add_49_grad/Reshape_1Reshapegradients_6/add_49_grad/Sum_1gradients_6/add_49_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
v
(gradients_6/add_49_grad/tuple/group_depsNoOp ^gradients_6/add_49_grad/Reshape"^gradients_6/add_49_grad/Reshape_1
о
0gradients_6/add_49_grad/tuple/control_dependencyIdentitygradients_6/add_49_grad/Reshape)^gradients_6/add_49_grad/tuple/group_deps*2
_class(
&$loc:@gradients_6/add_49_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_6/add_49_grad/tuple/control_dependency_1Identity!gradients_6/add_49_grad/Reshape_1)^gradients_6/add_49_grad/tuple/group_deps*
T0*
_output_shapes

:
*4
_class*
(&loc:@gradients_6/add_49_grad/Reshape_1
«
!gradients_6/MatMul_24_grad/MatMulMatMul0gradients_6/add_49_grad/tuple/control_dependencyVariable_48/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_6/MatMul_24_grad/MatMul_1MatMulTanh_140gradients_6/add_49_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
}
+gradients_6/MatMul_24_grad/tuple/group_depsNoOp"^gradients_6/MatMul_24_grad/MatMul$^gradients_6/MatMul_24_grad/MatMul_1
ш
3gradients_6/MatMul_24_grad/tuple/control_dependencyIdentity!gradients_6/MatMul_24_grad/MatMul,^gradients_6/MatMul_24_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*4
_class*
(&loc:@gradients_6/MatMul_24_grad/MatMul
х
5gradients_6/MatMul_24_grad/tuple/control_dependency_1Identity#gradients_6/MatMul_24_grad/MatMul_1,^gradients_6/MatMul_24_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_6/MatMul_24_grad/MatMul_1*
_output_shapes

:2

Э
!gradients_6/Tanh_14_grad/TanhGradTanhGradTanh_143gradients_6/MatMul_24_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
f
gradients_6/add_47_grad/ShapeShape	MatMul_23*
_output_shapes
:*
out_type0*
T0
p
gradients_6/add_47_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   2   
√
-gradients_6/add_47_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/add_47_grad/Shapegradients_6/add_47_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
gradients_6/add_47_grad/SumSum!gradients_6/Tanh_14_grad/TanhGrad-gradients_6/add_47_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_6/add_47_grad/ReshapeReshapegradients_6/add_47_grad/Sumgradients_6/add_47_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€2
Є
gradients_6/add_47_grad/Sum_1Sum!gradients_6/Tanh_14_grad/TanhGrad/gradients_6/add_47_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_6/add_47_grad/Reshape_1Reshapegradients_6/add_47_grad/Sum_1gradients_6/add_47_grad/Shape_1*
_output_shapes

:2*
Tshape0*
T0
v
(gradients_6/add_47_grad/tuple/group_depsNoOp ^gradients_6/add_47_grad/Reshape"^gradients_6/add_47_grad/Reshape_1
о
0gradients_6/add_47_grad/tuple/control_dependencyIdentitygradients_6/add_47_grad/Reshape)^gradients_6/add_47_grad/tuple/group_deps*2
_class(
&$loc:@gradients_6/add_47_grad/Reshape*'
_output_shapes
:€€€€€€€€€2*
T0
л
2gradients_6/add_47_grad/tuple/control_dependency_1Identity!gradients_6/add_47_grad/Reshape_1)^gradients_6/add_47_grad/tuple/group_deps*4
_class*
(&loc:@gradients_6/add_47_grad/Reshape_1*
_output_shapes

:2*
T0
«
!gradients_6/MatMul_23_grad/MatMulMatMul0gradients_6/add_47_grad/tuple/control_dependencyVariable_46/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_6/MatMul_23_grad/MatMul_1MatMulPlaceholder_410gradients_6/add_47_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
}
+gradients_6/MatMul_23_grad/tuple/group_depsNoOp"^gradients_6/MatMul_23_grad/MatMul$^gradients_6/MatMul_23_grad/MatMul_1
ш
3gradients_6/MatMul_23_grad/tuple/control_dependencyIdentity!gradients_6/MatMul_23_grad/MatMul,^gradients_6/MatMul_23_grad/tuple/group_deps*4
_class*
(&loc:@gradients_6/MatMul_23_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
х
5gradients_6/MatMul_23_grad/tuple/control_dependency_1Identity#gradients_6/MatMul_23_grad/MatMul_1,^gradients_6/MatMul_23_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_6/MatMul_23_grad/MatMul_1*
T0
d
GradientDescent_6/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
Т
9GradientDescent_6/update_Variable_46/ApplyGradientDescentApplyGradientDescentVariable_46GradientDescent_6/learning_rate5gradients_6/MatMul_23_grad/tuple/control_dependency_1*
_class
loc:@Variable_46*
_output_shapes

:@2*
T0*
use_locking( 
П
9GradientDescent_6/update_Variable_47/ApplyGradientDescentApplyGradientDescentVariable_47GradientDescent_6/learning_rate2gradients_6/add_47_grad/tuple/control_dependency_1*
_class
loc:@Variable_47*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_6/update_Variable_48/ApplyGradientDescentApplyGradientDescentVariable_48GradientDescent_6/learning_rate5gradients_6/MatMul_24_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:2
*
_class
loc:@Variable_48
П
9GradientDescent_6/update_Variable_49/ApplyGradientDescentApplyGradientDescentVariable_49GradientDescent_6/learning_rate2gradients_6/add_49_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_49*
_output_shapes

:

Й
GradientDescent_6NoOp:^GradientDescent_6/update_Variable_46/ApplyGradientDescent:^GradientDescent_6/update_Variable_47/ApplyGradientDescent:^GradientDescent_6/update_Variable_48/ApplyGradientDescent:^GradientDescent_6/update_Variable_49/ApplyGradientDescent
Г
Merge_5/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4l1/outputs_3l2/outputs_3loss_5l1/outputs_4l2/outputs_4loss_6*
_output_shapes
: *
N
Ь
init_2NoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign^Variable_44/Assign^Variable_45/Assign^Variable_46/Assign^Variable_47/Assign^Variable_48/Assign^Variable_49/Assign
Q
Placeholder_43Placeholder*
_output_shapes
:*
shape: *
dtype0
`
Placeholder_44Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
`
Placeholder_45Placeholder*'
_output_shapes
:€€€€€€€€€
*
shape: *
dtype0
g
random_normal_25/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Z
random_normal_25/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_25/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_25/RandomStandardNormalRandomStandardNormalrandom_normal_25/shape*
_output_shapes

:@2*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_25/mulMul%random_normal_25/RandomStandardNormalrandom_normal_25/stddev*
T0*
_output_shapes

:@2
m
random_normal_25Addrandom_normal_25/mulrandom_normal_25/mean*
T0*
_output_shapes

:@2

Variable_50
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
≠
Variable_50/AssignAssignVariable_50random_normal_25*
_class
loc:@Variable_50*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
r
Variable_50/readIdentityVariable_50*
T0*
_output_shapes

:@2*
_class
loc:@Variable_50
]
zeros_25Const*
_output_shapes

:2*
dtype0*
valueB2*    
M
add_50/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_50Addzeros_25add_50/y*
_output_shapes

:2*
T0

Variable_51
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_51/AssignAssignVariable_51add_50*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_51
r
Variable_51/readIdentityVariable_51*
_class
loc:@Variable_51*
_output_shapes

:2*
T0
Н
	MatMul_25MatMulPlaceholder_44Variable_50/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_51Add	MatMul_25Variable_51/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_23/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
I
Tanh_15Tanhadd_51*
T0*'
_output_shapes
:€€€€€€€€€2
]
l1/outputs_5/tagConst*
_output_shapes
: *
dtype0*
valueB Bl1/outputs_5
\
l1/outputs_5HistogramSummaryl1/outputs_5/tagTanh_15*
_output_shapes
: *
T0
g
random_normal_26/shapeConst*
dtype0*
_output_shapes
:*
valueB"2   
   
Z
random_normal_26/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_26/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_26/RandomStandardNormalRandomStandardNormalrandom_normal_26/shape*
dtype0*

seed *
T0*
_output_shapes

:2
*
seed2 
Д
random_normal_26/mulMul%random_normal_26/RandomStandardNormalrandom_normal_26/stddev*
T0*
_output_shapes

:2

m
random_normal_26Addrandom_normal_26/mulrandom_normal_26/mean*
T0*
_output_shapes

:2


Variable_52
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_52/AssignAssignVariable_52random_normal_26*
_output_shapes

:2
*
validate_shape(*
_class
loc:@Variable_52*
T0*
use_locking(
r
Variable_52/readIdentityVariable_52*
_class
loc:@Variable_52*
_output_shapes

:2
*
T0
]
zeros_26Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_52/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_52Addzeros_26add_52/y*
_output_shapes

:
*
T0

Variable_53
VariableV2*
shape
:
*
shared_name *
dtype0*
_output_shapes

:
*
	container 
£
Variable_53/AssignAssignVariable_53add_52*
_class
loc:@Variable_53*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
r
Variable_53/readIdentityVariable_53*
T0*
_output_shapes

:
*
_class
loc:@Variable_53
Ж
	MatMul_26MatMulTanh_15Variable_52/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_53Add	MatMul_26Variable_53/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_24/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
N
	Softmax_8Softmaxadd_53*'
_output_shapes
:€€€€€€€€€
*
T0
]
l2/outputs_5/tagConst*
valueB Bl2/outputs_5*
dtype0*
_output_shapes
: 
^
l2/outputs_5HistogramSummaryl2/outputs_5/tag	Softmax_8*
_output_shapes
: *
T0
I
Log_8Log	Softmax_8*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_8MulPlaceholder_45Log_8*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_8/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
w
Sum_8Summul_8Sum_8/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_8NegSum_8*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_8Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_8MeanNeg_8Const_8*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_7/tagsConst*
valueB Bloss_7*
_output_shapes
: *
dtype0
M
loss_7ScalarSummaryloss_7/tagsMean_8*
T0*
_output_shapes
: 
T
gradients_7/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_7/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
_
gradients_7/FillFillgradients_7/Shapegradients_7/Const*
T0*
_output_shapes
: 
o
%gradients_7/Mean_8_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_7/Mean_8_grad/ReshapeReshapegradients_7/Fill%gradients_7/Mean_8_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
b
gradients_7/Mean_8_grad/ShapeShapeNeg_8*
T0*
_output_shapes
:*
out_type0
§
gradients_7/Mean_8_grad/TileTilegradients_7/Mean_8_grad/Reshapegradients_7/Mean_8_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_7/Mean_8_grad/Shape_1ShapeNeg_8*
T0*
out_type0*
_output_shapes
:
b
gradients_7/Mean_8_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_7/Mean_8_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_7/Mean_8_grad/ProdProdgradients_7/Mean_8_grad/Shape_1gradients_7/Mean_8_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_7/Mean_8_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
¶
gradients_7/Mean_8_grad/Prod_1Prodgradients_7/Mean_8_grad/Shape_2gradients_7/Mean_8_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_7/Mean_8_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_7/Mean_8_grad/MaximumMaximumgradients_7/Mean_8_grad/Prod_1!gradients_7/Mean_8_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_7/Mean_8_grad/floordivFloorDivgradients_7/Mean_8_grad/Prodgradients_7/Mean_8_grad/Maximum*
_output_shapes
: *
T0
v
gradients_7/Mean_8_grad/CastCast gradients_7/Mean_8_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_7/Mean_8_grad/truedivRealDivgradients_7/Mean_8_grad/Tilegradients_7/Mean_8_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
p
gradients_7/Neg_8_grad/NegNeggradients_7/Mean_8_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_7/Sum_8_grad/ShapeShapemul_8*
T0*
out_type0*
_output_shapes
:
]
gradients_7/Sum_8_grad/SizeConst*
dtype0*
_output_shapes
: *
value	B :
|
gradients_7/Sum_8_grad/addAddSum_8/reduction_indicesgradients_7/Sum_8_grad/Size*
_output_shapes
:*
T0
Д
gradients_7/Sum_8_grad/modFloorModgradients_7/Sum_8_grad/addgradients_7/Sum_8_grad/Size*
_output_shapes
:*
T0
h
gradients_7/Sum_8_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
d
"gradients_7/Sum_8_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"gradients_7/Sum_8_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
≤
gradients_7/Sum_8_grad/rangeRange"gradients_7/Sum_8_grad/range/startgradients_7/Sum_8_grad/Size"gradients_7/Sum_8_grad/range/delta*

Tidx0*
_output_shapes
:
c
!gradients_7/Sum_8_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_7/Sum_8_grad/FillFillgradients_7/Sum_8_grad/Shape_1!gradients_7/Sum_8_grad/Fill/value*
T0*
_output_shapes
:
б
$gradients_7/Sum_8_grad/DynamicStitchDynamicStitchgradients_7/Sum_8_grad/rangegradients_7/Sum_8_grad/modgradients_7/Sum_8_grad/Shapegradients_7/Sum_8_grad/Fill*
N*
T0*#
_output_shapes
:€€€€€€€€€
b
 gradients_7/Sum_8_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
Я
gradients_7/Sum_8_grad/MaximumMaximum$gradients_7/Sum_8_grad/DynamicStitch gradients_7/Sum_8_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_7/Sum_8_grad/floordivFloorDivgradients_7/Sum_8_grad/Shapegradients_7/Sum_8_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_7/Sum_8_grad/ReshapeReshapegradients_7/Neg_8_grad/Neg$gradients_7/Sum_8_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
®
gradients_7/Sum_8_grad/TileTilegradients_7/Sum_8_grad/Reshapegradients_7/Sum_8_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_7/mul_8_grad/ShapeShapePlaceholder_45*
T0*
_output_shapes
:*
out_type0
c
gradients_7/mul_8_grad/Shape_1ShapeLog_8*
out_type0*
_output_shapes
:*
T0
ј
,gradients_7/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/mul_8_grad/Shapegradients_7/mul_8_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_7/mul_8_grad/mulMulgradients_7/Sum_8_grad/TileLog_8*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_7/mul_8_grad/SumSumgradients_7/mul_8_grad/mul,gradients_7/mul_8_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_7/mul_8_grad/ReshapeReshapegradients_7/mul_8_grad/Sumgradients_7/mul_8_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
В
gradients_7/mul_8_grad/mul_1MulPlaceholder_45gradients_7/Sum_8_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_7/mul_8_grad/Sum_1Sumgradients_7/mul_8_grad/mul_1.gradients_7/mul_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
©
 gradients_7/mul_8_grad/Reshape_1Reshapegradients_7/mul_8_grad/Sum_1gradients_7/mul_8_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

s
'gradients_7/mul_8_grad/tuple/group_depsNoOp^gradients_7/mul_8_grad/Reshape!^gradients_7/mul_8_grad/Reshape_1
к
/gradients_7/mul_8_grad/tuple/control_dependencyIdentitygradients_7/mul_8_grad/Reshape(^gradients_7/mul_8_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_7/mul_8_grad/Reshape
р
1gradients_7/mul_8_grad/tuple/control_dependency_1Identity gradients_7/mul_8_grad/Reshape_1(^gradients_7/mul_8_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*3
_class)
'%loc:@gradients_7/mul_8_grad/Reshape_1*
T0
†
!gradients_7/Log_8_grad/Reciprocal
Reciprocal	Softmax_82^gradients_7/mul_8_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

©
gradients_7/Log_8_grad/mulMul1gradients_7/mul_8_grad/tuple/control_dependency_1!gradients_7/Log_8_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_7/Softmax_8_grad/mulMulgradients_7/Log_8_grad/mul	Softmax_8*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_7/Softmax_8_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_7/Softmax_8_grad/SumSumgradients_7/Softmax_8_grad/mul0gradients_7/Softmax_8_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_7/Softmax_8_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
Ј
"gradients_7/Softmax_8_grad/ReshapeReshapegradients_7/Softmax_8_grad/Sum(gradients_7/Softmax_8_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ч
gradients_7/Softmax_8_grad/subSubgradients_7/Log_8_grad/mul"gradients_7/Softmax_8_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_7/Softmax_8_grad/mul_1Mulgradients_7/Softmax_8_grad/sub	Softmax_8*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_7/add_53_grad/ShapeShape	MatMul_26*
T0*
_output_shapes
:*
out_type0
p
gradients_7/add_53_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
√
-gradients_7/add_53_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/add_53_grad/Shapegradients_7/add_53_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_7/add_53_grad/SumSum gradients_7/Softmax_8_grad/mul_1-gradients_7/add_53_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_7/add_53_grad/ReshapeReshapegradients_7/add_53_grad/Sumgradients_7/add_53_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
Ј
gradients_7/add_53_grad/Sum_1Sum gradients_7/Softmax_8_grad/mul_1/gradients_7/add_53_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_7/add_53_grad/Reshape_1Reshapegradients_7/add_53_grad/Sum_1gradients_7/add_53_grad/Shape_1*
_output_shapes

:
*
Tshape0*
T0
v
(gradients_7/add_53_grad/tuple/group_depsNoOp ^gradients_7/add_53_grad/Reshape"^gradients_7/add_53_grad/Reshape_1
о
0gradients_7/add_53_grad/tuple/control_dependencyIdentitygradients_7/add_53_grad/Reshape)^gradients_7/add_53_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*2
_class(
&$loc:@gradients_7/add_53_grad/Reshape
л
2gradients_7/add_53_grad/tuple/control_dependency_1Identity!gradients_7/add_53_grad/Reshape_1)^gradients_7/add_53_grad/tuple/group_deps*
_output_shapes

:
*4
_class*
(&loc:@gradients_7/add_53_grad/Reshape_1*
T0
«
!gradients_7/MatMul_26_grad/MatMulMatMul0gradients_7/add_53_grad/tuple/control_dependencyVariable_52/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_7/MatMul_26_grad/MatMul_1MatMulTanh_150gradients_7/add_53_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
}
+gradients_7/MatMul_26_grad/tuple/group_depsNoOp"^gradients_7/MatMul_26_grad/MatMul$^gradients_7/MatMul_26_grad/MatMul_1
ш
3gradients_7/MatMul_26_grad/tuple/control_dependencyIdentity!gradients_7/MatMul_26_grad/MatMul,^gradients_7/MatMul_26_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_7/MatMul_26_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
х
5gradients_7/MatMul_26_grad/tuple/control_dependency_1Identity#gradients_7/MatMul_26_grad/MatMul_1,^gradients_7/MatMul_26_grad/tuple/group_deps*6
_class,
*(loc:@gradients_7/MatMul_26_grad/MatMul_1*
_output_shapes

:2
*
T0
Э
!gradients_7/Tanh_15_grad/TanhGradTanhGradTanh_153gradients_7/MatMul_26_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
f
gradients_7/add_51_grad/ShapeShape	MatMul_25*
T0*
out_type0*
_output_shapes
:
p
gradients_7/add_51_grad/Shape_1Const*
valueB"   2   *
_output_shapes
:*
dtype0
√
-gradients_7/add_51_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/add_51_grad/Shapegradients_7/add_51_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
gradients_7/add_51_grad/SumSum!gradients_7/Tanh_15_grad/TanhGrad-gradients_7/add_51_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_7/add_51_grad/ReshapeReshapegradients_7/add_51_grad/Sumgradients_7/add_51_grad/Shape*'
_output_shapes
:€€€€€€€€€2*
Tshape0*
T0
Є
gradients_7/add_51_grad/Sum_1Sum!gradients_7/Tanh_15_grad/TanhGrad/gradients_7/add_51_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_7/add_51_grad/Reshape_1Reshapegradients_7/add_51_grad/Sum_1gradients_7/add_51_grad/Shape_1*
Tshape0*
_output_shapes

:2*
T0
v
(gradients_7/add_51_grad/tuple/group_depsNoOp ^gradients_7/add_51_grad/Reshape"^gradients_7/add_51_grad/Reshape_1
о
0gradients_7/add_51_grad/tuple/control_dependencyIdentitygradients_7/add_51_grad/Reshape)^gradients_7/add_51_grad/tuple/group_deps*2
_class(
&$loc:@gradients_7/add_51_grad/Reshape*'
_output_shapes
:€€€€€€€€€2*
T0
л
2gradients_7/add_51_grad/tuple/control_dependency_1Identity!gradients_7/add_51_grad/Reshape_1)^gradients_7/add_51_grad/tuple/group_deps*
T0*
_output_shapes

:2*4
_class*
(&loc:@gradients_7/add_51_grad/Reshape_1
«
!gradients_7/MatMul_25_grad/MatMulMatMul0gradients_7/add_51_grad/tuple/control_dependencyVariable_50/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
T0
Њ
#gradients_7/MatMul_25_grad/MatMul_1MatMulPlaceholder_440gradients_7/add_51_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
}
+gradients_7/MatMul_25_grad/tuple/group_depsNoOp"^gradients_7/MatMul_25_grad/MatMul$^gradients_7/MatMul_25_grad/MatMul_1
ш
3gradients_7/MatMul_25_grad/tuple/control_dependencyIdentity!gradients_7/MatMul_25_grad/MatMul,^gradients_7/MatMul_25_grad/tuple/group_deps*4
_class*
(&loc:@gradients_7/MatMul_25_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
х
5gradients_7/MatMul_25_grad/tuple/control_dependency_1Identity#gradients_7/MatMul_25_grad/MatMul_1,^gradients_7/MatMul_25_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_7/MatMul_25_grad/MatMul_1*
T0
d
GradientDescent_7/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
Т
9GradientDescent_7/update_Variable_50/ApplyGradientDescentApplyGradientDescentVariable_50GradientDescent_7/learning_rate5gradients_7/MatMul_25_grad/tuple/control_dependency_1*
_class
loc:@Variable_50*
_output_shapes

:@2*
T0*
use_locking( 
П
9GradientDescent_7/update_Variable_51/ApplyGradientDescentApplyGradientDescentVariable_51GradientDescent_7/learning_rate2gradients_7/add_51_grad/tuple/control_dependency_1*
_output_shapes

:2*
_class
loc:@Variable_51*
T0*
use_locking( 
Т
9GradientDescent_7/update_Variable_52/ApplyGradientDescentApplyGradientDescentVariable_52GradientDescent_7/learning_rate5gradients_7/MatMul_26_grad/tuple/control_dependency_1*
_output_shapes

:2
*
_class
loc:@Variable_52*
T0*
use_locking( 
П
9GradientDescent_7/update_Variable_53/ApplyGradientDescentApplyGradientDescentVariable_53GradientDescent_7/learning_rate2gradients_7/add_53_grad/tuple/control_dependency_1*
_output_shapes

:
*
_class
loc:@Variable_53*
T0*
use_locking( 
Й
GradientDescent_7NoOp:^GradientDescent_7/update_Variable_50/ApplyGradientDescent:^GradientDescent_7/update_Variable_51/ApplyGradientDescent:^GradientDescent_7/update_Variable_52/ApplyGradientDescent:^GradientDescent_7/update_Variable_53/ApplyGradientDescent
І
Merge_6/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4l1/outputs_3l2/outputs_3loss_5l1/outputs_4l2/outputs_4loss_6l1/outputs_5l2/outputs_5loss_7*
_output_shapes
: *
N"ьr4з     Сwв`	‘CqQn]÷AJЏљ
≤К
9
Add
x"T
y"T
z"T"
Ttype:
2	
А
ApplyGradientDescent
var"TА

alpha"T

delta"T
out"TА"
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
S
HistogramSummary
tag
values"T
summary"
Ttype0:
2		
.
Identity

input"T
output"T"	
Ttype
+
Log
x"T
y"T"
Ttype:	
2
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	Р
К
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
Д
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
M
ScalarSummary
tags
values"T
summary"
Ttype:
2		
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И*1.0.02v1.0.0-rc2-15-g47bba63-dirtyду
]
PlaceholderPlaceholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
_
Placeholder_1Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
d
random_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ю
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
{
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes

:@2
d
random_normalAddrandom_normal/mulrandom_normal/mean*
_output_shapes

:@2*
T0
|
Variable
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
°
Variable/AssignAssignVariablerandom_normal*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable
i
Variable/readIdentityVariable*
_class
loc:@Variable*
_output_shapes

:@2*
T0
Z
zerosConst*
_output_shapes

:2*
dtype0*
valueB2*    
J
add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
A
addAddzerosadd/y*
_output_shapes

:2*
T0
~

Variable_1
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Э
Variable_1/AssignAssign
Variable_1add*
_class
loc:@Variable_1*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes

:2*
T0
Д
MatMulMatMulPlaceholderVariable/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
W
add_1AddMatMulVariable_1/read*'
_output_shapes
:€€€€€€€€€2*
T0
_
Placeholder_2Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
_
Placeholder_3Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

f
random_normal_1/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Y
random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_1/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
Ґ
$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*
_output_shapes

:@2*
seed2 *
T0*

seed *
dtype0
Б
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
_output_shapes

:@2*
T0
j
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes

:@2
~

Variable_2
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
©
Variable_2/AssignAssign
Variable_2random_normal_1*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_2*
T0*
use_locking(
o
Variable_2/readIdentity
Variable_2*
_output_shapes

:@2*
_class
loc:@Variable_2*
T0
\
zeros_1Const*
valueB2*    *
dtype0*
_output_shapes

:2
L
add_2/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
G
add_2Addzeros_1add_2/y*
T0*
_output_shapes

:2
~

Variable_3
VariableV2*
_output_shapes

:2*
	container *
dtype0*
shared_name *
shape
:2
Я
Variable_3/AssignAssign
Variable_3add_2*
_class
loc:@Variable_3*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
o
Variable_3/readIdentity
Variable_3*
T0*
_output_shapes

:2*
_class
loc:@Variable_3
К
MatMul_1MatMulPlaceholder_2Variable_2/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Y
add_3AddMatMul_1Variable_3/read*'
_output_shapes
:€€€€€€€€€2*
T0
V
dropout/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
E
TanhTanhadd_3*'
_output_shapes
:€€€€€€€€€2*
T0
_
Placeholder_4Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
_
Placeholder_5Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_2/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Y
random_normal_2/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_2/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
Ґ
$random_normal_2/RandomStandardNormalRandomStandardNormalrandom_normal_2/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_2/mulMul$random_normal_2/RandomStandardNormalrandom_normal_2/stddev*
_output_shapes

:@2*
T0
j
random_normal_2Addrandom_normal_2/mulrandom_normal_2/mean*
_output_shapes

:@2*
T0
~

Variable_4
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
©
Variable_4/AssignAssign
Variable_4random_normal_2*
_class
loc:@Variable_4*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
o
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4*
_output_shapes

:@2
\
zeros_2Const*
valueB2*    *
dtype0*
_output_shapes

:2
L
add_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
G
add_4Addzeros_2add_4/y*
_output_shapes

:2*
T0
~

Variable_5
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Я
Variable_5/AssignAssign
Variable_5add_4*
_class
loc:@Variable_5*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
o
Variable_5/readIdentity
Variable_5*
_output_shapes

:2*
_class
loc:@Variable_5*
T0
К
MatMul_2MatMulPlaceholder_4Variable_4/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Y
add_5AddMatMul_2Variable_5/read*'
_output_shapes
:€€€€€€€€€2*
T0
X
dropout_1/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
G
Tanh_1Tanhadd_5*
T0*'
_output_shapes
:€€€€€€€€€2
_
Placeholder_6Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
_
Placeholder_7Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

f
random_normal_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Y
random_normal_3/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
[
random_normal_3/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_3/RandomStandardNormalRandomStandardNormalrandom_normal_3/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_3/mulMul$random_normal_3/RandomStandardNormalrandom_normal_3/stddev*
_output_shapes

:@2*
T0
j
random_normal_3Addrandom_normal_3/mulrandom_normal_3/mean*
_output_shapes

:@2*
T0
~

Variable_6
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
©
Variable_6/AssignAssign
Variable_6random_normal_3*
_class
loc:@Variable_6*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
o
Variable_6/readIdentity
Variable_6*
_output_shapes

:@2*
_class
loc:@Variable_6*
T0
\
zeros_3Const*
dtype0*
_output_shapes

:2*
valueB2*    
L
add_6/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
G
add_6Addzeros_3add_6/y*
T0*
_output_shapes

:2
~

Variable_7
VariableV2*
_output_shapes

:2*
	container *
dtype0*
shared_name *
shape
:2
Я
Variable_7/AssignAssign
Variable_7add_6*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_7
o
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes

:2*
T0
К
MatMul_3MatMulPlaceholder_6Variable_6/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Y
add_7AddMatMul_3Variable_7/read*'
_output_shapes
:€€€€€€€€€2*
T0
_
Placeholder_8Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
_
Placeholder_9Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_4/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Y
random_normal_4/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
[
random_normal_4/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
Ґ
$random_normal_4/RandomStandardNormalRandomStandardNormalrandom_normal_4/shape*
_output_shapes

:@2*
seed2 *
dtype0*
T0*

seed 
Б
random_normal_4/mulMul$random_normal_4/RandomStandardNormalrandom_normal_4/stddev*
_output_shapes

:@2*
T0
j
random_normal_4Addrandom_normal_4/mulrandom_normal_4/mean*
T0*
_output_shapes

:@2
~

Variable_8
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
©
Variable_8/AssignAssign
Variable_8random_normal_4*
_class
loc:@Variable_8*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
o
Variable_8/readIdentity
Variable_8*
T0*
_output_shapes

:@2*
_class
loc:@Variable_8
\
zeros_4Const*
dtype0*
_output_shapes

:2*
valueB2*    
L
add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
G
add_8Addzeros_4add_8/y*
T0*
_output_shapes

:2
~

Variable_9
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
Я
Variable_9/AssignAssign
Variable_9add_8*
_class
loc:@Variable_9*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
o
Variable_9/readIdentity
Variable_9*
T0*
_output_shapes

:2*
_class
loc:@Variable_9
К
MatMul_4MatMulPlaceholder_8Variable_8/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Y
add_9AddMatMul_4Variable_9/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_2/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
G
Tanh_2Tanhadd_9*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_10Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_11Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
f
random_normal_5/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Y
random_normal_5/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_5/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_5/RandomStandardNormalRandomStandardNormalrandom_normal_5/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Б
random_normal_5/mulMul$random_normal_5/RandomStandardNormalrandom_normal_5/stddev*
_output_shapes

:@2*
T0
j
random_normal_5Addrandom_normal_5/mulrandom_normal_5/mean*
_output_shapes

:@2*
T0

Variable_10
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
ђ
Variable_10/AssignAssignVariable_10random_normal_5*
use_locking(*
T0*
_class
loc:@Variable_10*
validate_shape(*
_output_shapes

:@2
r
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10*
_output_shapes

:@2
\
zeros_5Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
I
add_10Addzeros_5add_10/y*
T0*
_output_shapes

:2

Variable_11
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_11/AssignAssignVariable_11add_10*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_11
r
Variable_11/readIdentityVariable_11*
T0*
_output_shapes

:2*
_class
loc:@Variable_11
М
MatMul_5MatMulPlaceholder_10Variable_10/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_11AddMatMul_5Variable_11/read*'
_output_shapes
:€€€€€€€€€2*
T0
X
dropout_3/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
H
Tanh_3Tanhadd_11*'
_output_shapes
:€€€€€€€€€2*
T0
`
Placeholder_12Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_13Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_6/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Y
random_normal_6/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_6/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ґ
$random_normal_6/RandomStandardNormalRandomStandardNormalrandom_normal_6/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_6/mulMul$random_normal_6/RandomStandardNormalrandom_normal_6/stddev*
T0*
_output_shapes

:@2
j
random_normal_6Addrandom_normal_6/mulrandom_normal_6/mean*
_output_shapes

:@2*
T0

Variable_12
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
ђ
Variable_12/AssignAssignVariable_12random_normal_6*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_12*
T0*
use_locking(
r
Variable_12/readIdentityVariable_12*
T0*
_output_shapes

:@2*
_class
loc:@Variable_12
\
zeros_6Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_12/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
I
add_12Addzeros_6add_12/y*
_output_shapes

:2*
T0

Variable_13
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_13/AssignAssignVariable_13add_12*
_class
loc:@Variable_13*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
r
Variable_13/readIdentityVariable_13*
_output_shapes

:2*
_class
loc:@Variable_13*
T0
М
MatMul_6MatMulPlaceholder_12Variable_12/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
[
add_13AddMatMul_6Variable_13/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_4/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
H
Tanh_4Tanhadd_13*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_14Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_15Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
f
random_normal_7/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Y
random_normal_7/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_7/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_7/RandomStandardNormalRandomStandardNormalrandom_normal_7/shape*
_output_shapes

:@2*
seed2 *
T0*

seed *
dtype0
Б
random_normal_7/mulMul$random_normal_7/RandomStandardNormalrandom_normal_7/stddev*
_output_shapes

:@2*
T0
j
random_normal_7Addrandom_normal_7/mulrandom_normal_7/mean*
_output_shapes

:@2*
T0

Variable_14
VariableV2*
_output_shapes

:@2*
	container *
dtype0*
shared_name *
shape
:@2
ђ
Variable_14/AssignAssignVariable_14random_normal_7*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_14*
T0*
use_locking(
r
Variable_14/readIdentityVariable_14*
_output_shapes

:@2*
_class
loc:@Variable_14*
T0
\
zeros_7Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_14/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
I
add_14Addzeros_7add_14/y*
T0*
_output_shapes

:2

Variable_15
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_15/AssignAssignVariable_15add_14*
use_locking(*
T0*
_class
loc:@Variable_15*
validate_shape(*
_output_shapes

:2
r
Variable_15/readIdentityVariable_15*
T0*
_class
loc:@Variable_15*
_output_shapes

:2
М
MatMul_7MatMulPlaceholder_14Variable_14/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_15AddMatMul_7Variable_15/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_5/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
H
Tanh_5Tanhadd_15*'
_output_shapes
:€€€€€€€€€2*
T0
`
Placeholder_16Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_17Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

f
random_normal_8/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Y
random_normal_8/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_8/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
Ґ
$random_normal_8/RandomStandardNormalRandomStandardNormalrandom_normal_8/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Б
random_normal_8/mulMul$random_normal_8/RandomStandardNormalrandom_normal_8/stddev*
_output_shapes

:@2*
T0
j
random_normal_8Addrandom_normal_8/mulrandom_normal_8/mean*
_output_shapes

:@2*
T0

Variable_16
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
ђ
Variable_16/AssignAssignVariable_16random_normal_8*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_16
r
Variable_16/readIdentityVariable_16*
T0*
_class
loc:@Variable_16*
_output_shapes

:@2
\
zeros_8Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_16/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
I
add_16Addzeros_8add_16/y*
_output_shapes

:2*
T0

Variable_17
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_17/AssignAssignVariable_17add_16*
_class
loc:@Variable_17*
_output_shapes

:2*
T0*
validate_shape(*
use_locking(
r
Variable_17/readIdentityVariable_17*
T0*
_output_shapes

:2*
_class
loc:@Variable_17
М
MatMul_8MatMulPlaceholder_16Variable_16/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
[
add_17AddMatMul_8Variable_17/read*
T0*'
_output_shapes
:€€€€€€€€€2
X
dropout_6/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
H
Tanh_6Tanhadd_17*'
_output_shapes
:€€€€€€€€€2*
T0
f
random_normal_9/shapeConst*
_output_shapes
:*
dtype0*
valueB"2   
   
Y
random_normal_9/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
[
random_normal_9/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ґ
$random_normal_9/RandomStandardNormalRandomStandardNormalrandom_normal_9/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Б
random_normal_9/mulMul$random_normal_9/RandomStandardNormalrandom_normal_9/stddev*
_output_shapes

:2
*
T0
j
random_normal_9Addrandom_normal_9/mulrandom_normal_9/mean*
T0*
_output_shapes

:2


Variable_18
VariableV2*
_output_shapes

:2
*
	container *
dtype0*
shared_name *
shape
:2

ђ
Variable_18/AssignAssignVariable_18random_normal_9*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2
*
_class
loc:@Variable_18
r
Variable_18/readIdentityVariable_18*
T0*
_output_shapes

:2
*
_class
loc:@Variable_18
\
zeros_9Const*
_output_shapes

:
*
dtype0*
valueB
*    
M
add_18/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
I
add_18Addzeros_9add_18/y*
T0*
_output_shapes

:


Variable_19
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
dtype0*
shared_name 
£
Variable_19/AssignAssignVariable_19add_18*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
*
_class
loc:@Variable_19
r
Variable_19/readIdentityVariable_19*
_class
loc:@Variable_19*
_output_shapes

:
*
T0
Д
MatMul_9MatMulTanh_6Variable_18/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
[
add_19AddMatMul_9Variable_19/read*'
_output_shapes
:€€€€€€€€€
*
T0
X
dropout_7/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
L
SoftmaxSoftmaxadd_19*
T0*'
_output_shapes
:€€€€€€€€€

E
LogLogSoftmax*
T0*'
_output_shapes
:€€€€€€€€€

Q
mulMulPlaceholder_17Log*'
_output_shapes
:€€€€€€€€€
*
T0
_
Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
q
SumSummulSum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
=
NegNegSum*#
_output_shapes
:€€€€€€€€€*
T0
O
ConstConst*
dtype0*
_output_shapes
:*
valueB: 
V
MeanMeanNegConst*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
`
Placeholder_18Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_19Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_10/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Z
random_normal_10/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_10/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_10/RandomStandardNormalRandomStandardNormalrandom_normal_10/shape*
_output_shapes

:@2*
seed2 *
T0*

seed *
dtype0
Д
random_normal_10/mulMul%random_normal_10/RandomStandardNormalrandom_normal_10/stddev*
T0*
_output_shapes

:@2
m
random_normal_10Addrandom_normal_10/mulrandom_normal_10/mean*
_output_shapes

:@2*
T0

Variable_20
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_20/AssignAssignVariable_20random_normal_10*
_class
loc:@Variable_20*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
r
Variable_20/readIdentityVariable_20*
T0*
_class
loc:@Variable_20*
_output_shapes

:@2
]
zeros_10Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_20/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_20Addzeros_10add_20/y*
T0*
_output_shapes

:2

Variable_21
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_21/AssignAssignVariable_21add_20*
_output_shapes

:2*
validate_shape(*
_class
loc:@Variable_21*
T0*
use_locking(
r
Variable_21/readIdentityVariable_21*
T0*
_output_shapes

:2*
_class
loc:@Variable_21
Н
	MatMul_10MatMulPlaceholder_18Variable_20/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_21Add	MatMul_10Variable_21/read*'
_output_shapes
:€€€€€€€€€2*
T0
X
dropout_8/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
H
Tanh_7Tanhadd_21*'
_output_shapes
:€€€€€€€€€2*
T0
g
random_normal_11/shapeConst*
_output_shapes
:*
dtype0*
valueB"2   
   
Z
random_normal_11/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_11/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_11/RandomStandardNormalRandomStandardNormalrandom_normal_11/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_11/mulMul%random_normal_11/RandomStandardNormalrandom_normal_11/stddev*
T0*
_output_shapes

:2

m
random_normal_11Addrandom_normal_11/mulrandom_normal_11/mean*
T0*
_output_shapes

:2


Variable_22
VariableV2*
_output_shapes

:2
*
	container *
shape
:2
*
dtype0*
shared_name 
≠
Variable_22/AssignAssignVariable_22random_normal_11*
_output_shapes

:2
*
validate_shape(*
_class
loc:@Variable_22*
T0*
use_locking(
r
Variable_22/readIdentityVariable_22*
T0*
_output_shapes

:2
*
_class
loc:@Variable_22
]
zeros_11Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_22/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_22Addzeros_11add_22/y*
_output_shapes

:
*
T0

Variable_23
VariableV2*
_output_shapes

:
*
	container *
dtype0*
shared_name *
shape
:

£
Variable_23/AssignAssignVariable_23add_22*
_output_shapes

:
*
validate_shape(*
_class
loc:@Variable_23*
T0*
use_locking(
r
Variable_23/readIdentityVariable_23*
_output_shapes

:
*
_class
loc:@Variable_23*
T0
Е
	MatMul_11MatMulTanh_7Variable_22/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
\
add_23Add	MatMul_11Variable_23/read*
T0*'
_output_shapes
:€€€€€€€€€

X
dropout_9/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
N
	Softmax_1Softmaxadd_23*'
_output_shapes
:€€€€€€€€€
*
T0
I
Log_1Log	Softmax_1*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_1MulPlaceholder_19Log_1*
T0*'
_output_shapes
:€€€€€€€€€

a
Sum_1/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
w
Sum_1Summul_1Sum_1/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_1NegSum_1*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_1MeanNeg_1Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
I
lossScalarSummary	loss/tagsMean_1*
_output_shapes
: *
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
T
gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Y
gradients/FillFillgradients/Shapegradients/Const*
T0*
_output_shapes
: 
m
#gradients/Mean_1_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Р
gradients/Mean_1_grad/ReshapeReshapegradients/Fill#gradients/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
`
gradients/Mean_1_grad/ShapeShapeNeg_1*
T0*
out_type0*
_output_shapes
:
Ю
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
b
gradients/Mean_1_grad/Shape_1ShapeNeg_1*
T0*
out_type0*
_output_shapes
:
`
gradients/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
e
gradients/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
Ь
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
g
gradients/Mean_1_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
†
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
И
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
Ж
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
_output_shapes
: *
T0
r
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
О
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
l
gradients/Neg_1_grad/NegNeggradients/Mean_1_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
_
gradients/Sum_1_grad/ShapeShapemul_1*
T0*
out_type0*
_output_shapes
:
[
gradients/Sum_1_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
x
gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
_output_shapes
:*
T0
~
gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_output_shapes
:*
T0
f
gradients/Sum_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
b
 gradients/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
b
 gradients/Sum_1_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
™
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*

Tidx0*
_output_shapes
:
a
gradients/Sum_1_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Е
gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_output_shapes
:*
T0
„
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
`
gradients/Sum_1_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Щ
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
И
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_output_shapes
:*
T0
Ц
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_1_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0
Ґ
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

h
gradients/mul_1_grad/ShapeShapePlaceholder_19*
T0*
out_type0*
_output_shapes
:
a
gradients/mul_1_grad/Shape_1ShapeLog_1*
T0*
_output_shapes
:*
out_type0
Ї
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
s
gradients/mul_1_grad/mulMulgradients/Sum_1_grad/TileLog_1*
T0*'
_output_shapes
:€€€€€€€€€

•
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
~
gradients/mul_1_grad/mul_1MulPlaceholder_19gradients/Sum_1_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
m
%gradients/mul_1_grad/tuple/group_depsNoOp^gradients/mul_1_grad/Reshape^gradients/mul_1_grad/Reshape_1
в
-gradients/mul_1_grad/tuple/control_dependencyIdentitygradients/mul_1_grad/Reshape&^gradients/mul_1_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*/
_class%
#!loc:@gradients/mul_1_grad/Reshape
и
/gradients/mul_1_grad/tuple/control_dependency_1Identitygradients/mul_1_grad/Reshape_1&^gradients/mul_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_1_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
*
T0
Ь
gradients/Log_1_grad/Reciprocal
Reciprocal	Softmax_10^gradients/mul_1_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
£
gradients/Log_1_grad/mulMul/gradients/mul_1_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

z
gradients/Softmax_1_grad/mulMulgradients/Log_1_grad/mul	Softmax_1*
T0*'
_output_shapes
:€€€€€€€€€

x
.gradients/Softmax_1_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Љ
gradients/Softmax_1_grad/SumSumgradients/Softmax_1_grad/mul.gradients/Softmax_1_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
w
&gradients/Softmax_1_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
±
 gradients/Softmax_1_grad/ReshapeReshapegradients/Softmax_1_grad/Sum&gradients/Softmax_1_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
С
gradients/Softmax_1_grad/subSubgradients/Log_1_grad/mul gradients/Softmax_1_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

А
gradients/Softmax_1_grad/mul_1Mulgradients/Softmax_1_grad/sub	Softmax_1*
T0*'
_output_shapes
:€€€€€€€€€

d
gradients/add_23_grad/ShapeShape	MatMul_11*
_output_shapes
:*
out_type0*
T0
n
gradients/add_23_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
љ
+gradients/add_23_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_23_grad/Shapegradients/add_23_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≠
gradients/add_23_grad/SumSumgradients/Softmax_1_grad/mul_1+gradients/add_23_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
†
gradients/add_23_grad/ReshapeReshapegradients/add_23_grad/Sumgradients/add_23_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
±
gradients/add_23_grad/Sum_1Sumgradients/Softmax_1_grad/mul_1-gradients/add_23_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Э
gradients/add_23_grad/Reshape_1Reshapegradients/add_23_grad/Sum_1gradients/add_23_grad/Shape_1*
T0*
_output_shapes

:
*
Tshape0
p
&gradients/add_23_grad/tuple/group_depsNoOp^gradients/add_23_grad/Reshape ^gradients/add_23_grad/Reshape_1
ж
.gradients/add_23_grad/tuple/control_dependencyIdentitygradients/add_23_grad/Reshape'^gradients/add_23_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*0
_class&
$"loc:@gradients/add_23_grad/Reshape*
T0
г
0gradients/add_23_grad/tuple/control_dependency_1Identitygradients/add_23_grad/Reshape_1'^gradients/add_23_grad/tuple/group_deps*
T0*
_output_shapes

:
*2
_class(
&$loc:@gradients/add_23_grad/Reshape_1
√
gradients/MatMul_11_grad/MatMulMatMul.gradients/add_23_grad/tuple/control_dependencyVariable_22/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
≤
!gradients/MatMul_11_grad/MatMul_1MatMulTanh_7.gradients/add_23_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
w
)gradients/MatMul_11_grad/tuple/group_depsNoOp ^gradients/MatMul_11_grad/MatMul"^gradients/MatMul_11_grad/MatMul_1
р
1gradients/MatMul_11_grad/tuple/control_dependencyIdentitygradients/MatMul_11_grad/MatMul*^gradients/MatMul_11_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/MatMul_11_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
н
3gradients/MatMul_11_grad/tuple/control_dependency_1Identity!gradients/MatMul_11_grad/MatMul_1*^gradients/MatMul_11_grad/tuple/group_deps*4
_class*
(&loc:@gradients/MatMul_11_grad/MatMul_1*
_output_shapes

:2
*
T0
Ч
gradients/Tanh_7_grad/TanhGradTanhGradTanh_71gradients/MatMul_11_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
d
gradients/add_21_grad/ShapeShape	MatMul_10*
T0*
out_type0*
_output_shapes
:
n
gradients/add_21_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   2   
љ
+gradients/add_21_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_21_grad/Shapegradients/add_21_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≠
gradients/add_21_grad/SumSumgradients/Tanh_7_grad/TanhGrad+gradients/add_21_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
†
gradients/add_21_grad/ReshapeReshapegradients/add_21_grad/Sumgradients/add_21_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€2*
T0
±
gradients/add_21_grad/Sum_1Sumgradients/Tanh_7_grad/TanhGrad-gradients/add_21_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Э
gradients/add_21_grad/Reshape_1Reshapegradients/add_21_grad/Sum_1gradients/add_21_grad/Shape_1*
T0*
_output_shapes

:2*
Tshape0
p
&gradients/add_21_grad/tuple/group_depsNoOp^gradients/add_21_grad/Reshape ^gradients/add_21_grad/Reshape_1
ж
.gradients/add_21_grad/tuple/control_dependencyIdentitygradients/add_21_grad/Reshape'^gradients/add_21_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*0
_class&
$"loc:@gradients/add_21_grad/Reshape*
T0
г
0gradients/add_21_grad/tuple/control_dependency_1Identitygradients/add_21_grad/Reshape_1'^gradients/add_21_grad/tuple/group_deps*
T0*
_output_shapes

:2*2
_class(
&$loc:@gradients/add_21_grad/Reshape_1
√
gradients/MatMul_10_grad/MatMulMatMul.gradients/add_21_grad/tuple/control_dependencyVariable_20/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Ї
!gradients/MatMul_10_grad/MatMul_1MatMulPlaceholder_18.gradients/add_21_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
w
)gradients/MatMul_10_grad/tuple/group_depsNoOp ^gradients/MatMul_10_grad/MatMul"^gradients/MatMul_10_grad/MatMul_1
р
1gradients/MatMul_10_grad/tuple/control_dependencyIdentitygradients/MatMul_10_grad/MatMul*^gradients/MatMul_10_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/MatMul_10_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
н
3gradients/MatMul_10_grad/tuple/control_dependency_1Identity!gradients/MatMul_10_grad/MatMul_1*^gradients/MatMul_10_grad/tuple/group_deps*
T0*
_output_shapes

:@2*4
_class*
(&loc:@gradients/MatMul_10_grad/MatMul_1
b
GradientDescent/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
М
7GradientDescent/update_Variable_20/ApplyGradientDescentApplyGradientDescentVariable_20GradientDescent/learning_rate3gradients/MatMul_10_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:@2*
_class
loc:@Variable_20
Й
7GradientDescent/update_Variable_21/ApplyGradientDescentApplyGradientDescentVariable_21GradientDescent/learning_rate0gradients/add_21_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:2*
_class
loc:@Variable_21
М
7GradientDescent/update_Variable_22/ApplyGradientDescentApplyGradientDescentVariable_22GradientDescent/learning_rate3gradients/MatMul_11_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:2
*
_class
loc:@Variable_22
Й
7GradientDescent/update_Variable_23/ApplyGradientDescentApplyGradientDescentVariable_23GradientDescent/learning_rate0gradients/add_23_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:
*
_class
loc:@Variable_23
€
GradientDescentNoOp8^GradientDescent/update_Variable_20/ApplyGradientDescent8^GradientDescent/update_Variable_21/ApplyGradientDescent8^GradientDescent/update_Variable_22/ApplyGradientDescent8^GradientDescent/update_Variable_23/ApplyGradientDescent
Q
Placeholder_20Placeholder*
_output_shapes
:*
dtype0*
shape: 
`
Placeholder_21Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_22Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

g
random_normal_12/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Z
random_normal_12/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_12/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_12/RandomStandardNormalRandomStandardNormalrandom_normal_12/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_12/mulMul%random_normal_12/RandomStandardNormalrandom_normal_12/stddev*
_output_shapes

:@2*
T0
m
random_normal_12Addrandom_normal_12/mulrandom_normal_12/mean*
T0*
_output_shapes

:@2

Variable_24
VariableV2*
_output_shapes

:@2*
	container *
shape
:@2*
dtype0*
shared_name 
≠
Variable_24/AssignAssignVariable_24random_normal_12*
use_locking(*
T0*
_class
loc:@Variable_24*
validate_shape(*
_output_shapes

:@2
r
Variable_24/readIdentityVariable_24*
T0*
_class
loc:@Variable_24*
_output_shapes

:@2
]
zeros_12Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_24/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_24Addzeros_12add_24/y*
T0*
_output_shapes

:2

Variable_25
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_25/AssignAssignVariable_25add_24*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_25
r
Variable_25/readIdentityVariable_25*
T0*
_output_shapes

:2*
_class
loc:@Variable_25
Н
	MatMul_12MatMulPlaceholder_21Variable_24/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_25Add	MatMul_12Variable_25/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_10/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
H
Tanh_8Tanhadd_25*'
_output_shapes
:€€€€€€€€€2*
T0
g
random_normal_13/shapeConst*
valueB"2   
   *
dtype0*
_output_shapes
:
Z
random_normal_13/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_13/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
§
%random_normal_13/RandomStandardNormalRandomStandardNormalrandom_normal_13/shape*
dtype0*

seed *
T0*
_output_shapes

:2
*
seed2 
Д
random_normal_13/mulMul%random_normal_13/RandomStandardNormalrandom_normal_13/stddev*
_output_shapes

:2
*
T0
m
random_normal_13Addrandom_normal_13/mulrandom_normal_13/mean*
_output_shapes

:2
*
T0

Variable_26
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_26/AssignAssignVariable_26random_normal_13*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2
*
_class
loc:@Variable_26
r
Variable_26/readIdentityVariable_26*
_output_shapes

:2
*
_class
loc:@Variable_26*
T0
]
zeros_13Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_26/yConst*
valueB
 *Ќћћ=*
dtype0*
_output_shapes
: 
J
add_26Addzeros_13add_26/y*
T0*
_output_shapes

:


Variable_27
VariableV2*
_output_shapes

:
*
	container *
dtype0*
shared_name *
shape
:

£
Variable_27/AssignAssignVariable_27add_26*
_class
loc:@Variable_27*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
r
Variable_27/readIdentityVariable_27*
T0*
_class
loc:@Variable_27*
_output_shapes

:

Е
	MatMul_13MatMulTanh_8Variable_26/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_27Add	MatMul_13Variable_27/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_11/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
N
	Softmax_2Softmaxadd_27*'
_output_shapes
:€€€€€€€€€
*
T0
I
Log_2Log	Softmax_2*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_2MulPlaceholder_22Log_2*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_2/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
w
Sum_2Summul_2Sum_2/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_2NegSum_2*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_2MeanNeg_2Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
loss_1/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_1
M
loss_1ScalarSummaryloss_1/tagsMean_2*
_output_shapes
: *
T0
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_1/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0*
_output_shapes
: 
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Ц
gradients_1/Mean_2_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_2_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients_1/Mean_2_grad/ShapeShapeNeg_2*
_output_shapes
:*
out_type0*
T0
§
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
d
gradients_1/Mean_2_grad/Shape_1ShapeNeg_2*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_2_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_1/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ґ
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
¶
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_1/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
О
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
_output_shapes
: *
T0
v
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_1/Neg_2_grad/NegNeggradients_1/Mean_2_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_1/Sum_2_grad/ShapeShapemul_2*
_output_shapes
:*
out_type0*
T0
]
gradients_1/Sum_2_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
gradients_1/Sum_2_grad/addAddSum_2/reduction_indicesgradients_1/Sum_2_grad/Size*
_output_shapes
:*
T0
Д
gradients_1/Sum_2_grad/modFloorModgradients_1/Sum_2_grad/addgradients_1/Sum_2_grad/Size*
_output_shapes
:*
T0
h
gradients_1/Sum_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
d
"gradients_1/Sum_2_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"gradients_1/Sum_2_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
≤
gradients_1/Sum_2_grad/rangeRange"gradients_1/Sum_2_grad/range/startgradients_1/Sum_2_grad/Size"gradients_1/Sum_2_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_1/Sum_2_grad/Fill/valueConst*
_output_shapes
: *
dtype0*
value	B :
Л
gradients_1/Sum_2_grad/FillFillgradients_1/Sum_2_grad/Shape_1!gradients_1/Sum_2_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_1/Sum_2_grad/DynamicStitchDynamicStitchgradients_1/Sum_2_grad/rangegradients_1/Sum_2_grad/modgradients_1/Sum_2_grad/Shapegradients_1/Sum_2_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
b
 gradients_1/Sum_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
Я
gradients_1/Sum_2_grad/MaximumMaximum$gradients_1/Sum_2_grad/DynamicStitch gradients_1/Sum_2_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
О
gradients_1/Sum_2_grad/floordivFloorDivgradients_1/Sum_2_grad/Shapegradients_1/Sum_2_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_1/Sum_2_grad/ReshapeReshapegradients_1/Neg_2_grad/Neg$gradients_1/Sum_2_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
®
gradients_1/Sum_2_grad/TileTilegradients_1/Sum_2_grad/Reshapegradients_1/Sum_2_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_1/mul_2_grad/ShapeShapePlaceholder_22*
_output_shapes
:*
out_type0*
T0
c
gradients_1/mul_2_grad/Shape_1ShapeLog_2*
_output_shapes
:*
out_type0*
T0
ј
,gradients_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_2_grad/Shapegradients_1/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
w
gradients_1/mul_2_grad/mulMulgradients_1/Sum_2_grad/TileLog_2*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients_1/mul_2_grad/SumSumgradients_1/mul_2_grad/mul,gradients_1/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_1/mul_2_grad/ReshapeReshapegradients_1/mul_2_grad/Sumgradients_1/mul_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

В
gradients_1/mul_2_grad/mul_1MulPlaceholder_22gradients_1/Sum_2_grad/Tile*'
_output_shapes
:€€€€€€€€€
*
T0
±
gradients_1/mul_2_grad/Sum_1Sumgradients_1/mul_2_grad/mul_1.gradients_1/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_1/mul_2_grad/Reshape_1Reshapegradients_1/mul_2_grad/Sum_1gradients_1/mul_2_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
s
'gradients_1/mul_2_grad/tuple/group_depsNoOp^gradients_1/mul_2_grad/Reshape!^gradients_1/mul_2_grad/Reshape_1
к
/gradients_1/mul_2_grad/tuple/control_dependencyIdentitygradients_1/mul_2_grad/Reshape(^gradients_1/mul_2_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_1/mul_2_grad/Reshape
р
1gradients_1/mul_2_grad/tuple/control_dependency_1Identity gradients_1/mul_2_grad/Reshape_1(^gradients_1/mul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/mul_2_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_1/Log_2_grad/Reciprocal
Reciprocal	Softmax_22^gradients_1/mul_2_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_1/Log_2_grad/mulMul1gradients_1/mul_2_grad/tuple/control_dependency_1!gradients_1/Log_2_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_1/Softmax_2_grad/mulMulgradients_1/Log_2_grad/mul	Softmax_2*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_1/Softmax_2_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_1/Softmax_2_grad/SumSumgradients_1/Softmax_2_grad/mul0gradients_1/Softmax_2_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_1/Softmax_2_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
Ј
"gradients_1/Softmax_2_grad/ReshapeReshapegradients_1/Softmax_2_grad/Sum(gradients_1/Softmax_2_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ч
gradients_1/Softmax_2_grad/subSubgradients_1/Log_2_grad/mul"gradients_1/Softmax_2_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_1/Softmax_2_grad/mul_1Mulgradients_1/Softmax_2_grad/sub	Softmax_2*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_1/add_27_grad/ShapeShape	MatMul_13*
T0*
out_type0*
_output_shapes
:
p
gradients_1/add_27_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
√
-gradients_1/add_27_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_27_grad/Shapegradients_1/add_27_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_1/add_27_grad/SumSum gradients_1/Softmax_2_grad/mul_1-gradients_1/add_27_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_1/add_27_grad/ReshapeReshapegradients_1/add_27_grad/Sumgradients_1/add_27_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
Ј
gradients_1/add_27_grad/Sum_1Sum gradients_1/Softmax_2_grad/mul_1/gradients_1/add_27_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_1/add_27_grad/Reshape_1Reshapegradients_1/add_27_grad/Sum_1gradients_1/add_27_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
v
(gradients_1/add_27_grad/tuple/group_depsNoOp ^gradients_1/add_27_grad/Reshape"^gradients_1/add_27_grad/Reshape_1
о
0gradients_1/add_27_grad/tuple/control_dependencyIdentitygradients_1/add_27_grad/Reshape)^gradients_1/add_27_grad/tuple/group_deps*2
_class(
&$loc:@gradients_1/add_27_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_1/add_27_grad/tuple/control_dependency_1Identity!gradients_1/add_27_grad/Reshape_1)^gradients_1/add_27_grad/tuple/group_deps*
_output_shapes

:
*4
_class*
(&loc:@gradients_1/add_27_grad/Reshape_1*
T0
«
!gradients_1/MatMul_13_grad/MatMulMatMul0gradients_1/add_27_grad/tuple/control_dependencyVariable_26/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
ґ
#gradients_1/MatMul_13_grad/MatMul_1MatMulTanh_80gradients_1/add_27_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_1/MatMul_13_grad/tuple/group_depsNoOp"^gradients_1/MatMul_13_grad/MatMul$^gradients_1/MatMul_13_grad/MatMul_1
ш
3gradients_1/MatMul_13_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_13_grad/MatMul,^gradients_1/MatMul_13_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/MatMul_13_grad/MatMul*'
_output_shapes
:€€€€€€€€€2*
T0
х
5gradients_1/MatMul_13_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_13_grad/MatMul_1,^gradients_1/MatMul_13_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_1/MatMul_13_grad/MatMul_1*
_output_shapes

:2

Ы
 gradients_1/Tanh_8_grad/TanhGradTanhGradTanh_83gradients_1/MatMul_13_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_1/add_25_grad/ShapeShape	MatMul_12*
T0*
_output_shapes
:*
out_type0
p
gradients_1/add_25_grad/Shape_1Const*
valueB"   2   *
dtype0*
_output_shapes
:
√
-gradients_1/add_25_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_25_grad/Shapegradients_1/add_25_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_1/add_25_grad/SumSum gradients_1/Tanh_8_grad/TanhGrad-gradients_1/add_25_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_1/add_25_grad/ReshapeReshapegradients_1/add_25_grad/Sumgradients_1/add_25_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€2*
T0
Ј
gradients_1/add_25_grad/Sum_1Sum gradients_1/Tanh_8_grad/TanhGrad/gradients_1/add_25_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_1/add_25_grad/Reshape_1Reshapegradients_1/add_25_grad/Sum_1gradients_1/add_25_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:2
v
(gradients_1/add_25_grad/tuple/group_depsNoOp ^gradients_1/add_25_grad/Reshape"^gradients_1/add_25_grad/Reshape_1
о
0gradients_1/add_25_grad/tuple/control_dependencyIdentitygradients_1/add_25_grad/Reshape)^gradients_1/add_25_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_1/add_25_grad/Reshape
л
2gradients_1/add_25_grad/tuple/control_dependency_1Identity!gradients_1/add_25_grad/Reshape_1)^gradients_1/add_25_grad/tuple/group_deps*
T0*
_output_shapes

:2*4
_class*
(&loc:@gradients_1/add_25_grad/Reshape_1
«
!gradients_1/MatMul_12_grad/MatMulMatMul0gradients_1/add_25_grad/tuple/control_dependencyVariable_24/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_1/MatMul_12_grad/MatMul_1MatMulPlaceholder_210gradients_1/add_25_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:@2*
transpose_a(*
T0
}
+gradients_1/MatMul_12_grad/tuple/group_depsNoOp"^gradients_1/MatMul_12_grad/MatMul$^gradients_1/MatMul_12_grad/MatMul_1
ш
3gradients_1/MatMul_12_grad/tuple/control_dependencyIdentity!gradients_1/MatMul_12_grad/MatMul,^gradients_1/MatMul_12_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/MatMul_12_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
х
5gradients_1/MatMul_12_grad/tuple/control_dependency_1Identity#gradients_1/MatMul_12_grad/MatMul_1,^gradients_1/MatMul_12_grad/tuple/group_deps*6
_class,
*(loc:@gradients_1/MatMul_12_grad/MatMul_1*
_output_shapes

:@2*
T0
d
GradientDescent_1/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
Т
9GradientDescent_1/update_Variable_24/ApplyGradientDescentApplyGradientDescentVariable_24GradientDescent_1/learning_rate5gradients_1/MatMul_12_grad/tuple/control_dependency_1*
_output_shapes

:@2*
_class
loc:@Variable_24*
T0*
use_locking( 
П
9GradientDescent_1/update_Variable_25/ApplyGradientDescentApplyGradientDescentVariable_25GradientDescent_1/learning_rate2gradients_1/add_25_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_25*
_output_shapes

:2
Т
9GradientDescent_1/update_Variable_26/ApplyGradientDescentApplyGradientDescentVariable_26GradientDescent_1/learning_rate5gradients_1/MatMul_13_grad/tuple/control_dependency_1*
_output_shapes

:2
*
_class
loc:@Variable_26*
T0*
use_locking( 
П
9GradientDescent_1/update_Variable_27/ApplyGradientDescentApplyGradientDescentVariable_27GradientDescent_1/learning_rate2gradients_1/add_27_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_27*
_output_shapes

:

Й
GradientDescent_1NoOp:^GradientDescent_1/update_Variable_24/ApplyGradientDescent:^GradientDescent_1/update_Variable_25/ApplyGradientDescent:^GradientDescent_1/update_Variable_26/ApplyGradientDescent:^GradientDescent_1/update_Variable_27/ApplyGradientDescent
Q
Placeholder_23Placeholder*
shape: *
dtype0*
_output_shapes
:
Q
Merge/MergeSummaryMergeSummarylossloss_1*
N*
_output_shapes
: 
`
Placeholder_24Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_25Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

g
random_normal_14/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Z
random_normal_14/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_14/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_14/RandomStandardNormalRandomStandardNormalrandom_normal_14/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_14/mulMul%random_normal_14/RandomStandardNormalrandom_normal_14/stddev*
_output_shapes

:@2*
T0
m
random_normal_14Addrandom_normal_14/mulrandom_normal_14/mean*
T0*
_output_shapes

:@2

Variable_28
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_28/AssignAssignVariable_28random_normal_14*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_28
r
Variable_28/readIdentityVariable_28*
T0*
_output_shapes

:@2*
_class
loc:@Variable_28
]
zeros_14Const*
valueB2*    *
_output_shapes

:2*
dtype0
M
add_28/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
J
add_28Addzeros_14add_28/y*
_output_shapes

:2*
T0

Variable_29
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_29/AssignAssignVariable_29add_28*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_29
r
Variable_29/readIdentityVariable_29*
T0*
_output_shapes

:2*
_class
loc:@Variable_29
Н
	MatMul_14MatMulPlaceholder_24Variable_28/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_29Add	MatMul_14Variable_29/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_12/keep_probConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
H
Tanh_9Tanhadd_29*
T0*'
_output_shapes
:€€€€€€€€€2
`
Placeholder_26Placeholder*'
_output_shapes
:€€€€€€€€€@*
dtype0*
shape: 
`
Placeholder_27Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_15/shapeConst*
valueB"@   2   *
_output_shapes
:*
dtype0
Z
random_normal_15/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_15/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
§
%random_normal_15/RandomStandardNormalRandomStandardNormalrandom_normal_15/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_15/mulMul%random_normal_15/RandomStandardNormalrandom_normal_15/stddev*
_output_shapes

:@2*
T0
m
random_normal_15Addrandom_normal_15/mulrandom_normal_15/mean*
T0*
_output_shapes

:@2

Variable_30
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_30/AssignAssignVariable_30random_normal_15*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_30
r
Variable_30/readIdentityVariable_30*
T0*
_output_shapes

:@2*
_class
loc:@Variable_30
]
zeros_15Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_30/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
J
add_30Addzeros_15add_30/y*
_output_shapes

:2*
T0

Variable_31
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_31/AssignAssignVariable_31add_30*
use_locking(*
T0*
_class
loc:@Variable_31*
validate_shape(*
_output_shapes

:2
r
Variable_31/readIdentityVariable_31*
T0*
_class
loc:@Variable_31*
_output_shapes

:2
Н
	MatMul_15MatMulPlaceholder_26Variable_30/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_31Add	MatMul_15Variable_31/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_13/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
I
Tanh_10Tanhadd_31*
T0*'
_output_shapes
:€€€€€€€€€2
Y
l1/outputs/tagConst*
valueB B
l1/outputs*
_output_shapes
: *
dtype0
X

l1/outputsHistogramSummaryl1/outputs/tagTanh_10*
_output_shapes
: *
T0
g
random_normal_16/shapeConst*
dtype0*
_output_shapes
:*
valueB"2   
   
Z
random_normal_16/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_16/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_16/RandomStandardNormalRandomStandardNormalrandom_normal_16/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_16/mulMul%random_normal_16/RandomStandardNormalrandom_normal_16/stddev*
T0*
_output_shapes

:2

m
random_normal_16Addrandom_normal_16/mulrandom_normal_16/mean*
_output_shapes

:2
*
T0

Variable_32
VariableV2*
shape
:2
*
shared_name *
dtype0*
_output_shapes

:2
*
	container 
≠
Variable_32/AssignAssignVariable_32random_normal_16*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2
*
_class
loc:@Variable_32
r
Variable_32/readIdentityVariable_32*
_output_shapes

:2
*
_class
loc:@Variable_32*
T0
]
zeros_16Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_32/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
J
add_32Addzeros_16add_32/y*
_output_shapes

:
*
T0

Variable_33
VariableV2*
_output_shapes

:
*
	container *
shape
:
*
dtype0*
shared_name 
£
Variable_33/AssignAssignVariable_33add_32*
_class
loc:@Variable_33*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
r
Variable_33/readIdentityVariable_33*
T0*
_output_shapes

:
*
_class
loc:@Variable_33
Ж
	MatMul_16MatMulTanh_10Variable_32/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_33Add	MatMul_16Variable_33/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_14/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
N
	Softmax_3Softmaxadd_33*
T0*'
_output_shapes
:€€€€€€€€€

Y
l2/outputs/tagConst*
valueB B
l2/outputs*
dtype0*
_output_shapes
: 
Z

l2/outputsHistogramSummaryl2/outputs/tag	Softmax_3*
T0*
_output_shapes
: 
I
Log_3Log	Softmax_3*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_3MulPlaceholder_27Log_3*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_3/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
w
Sum_3Summul_3Sum_3/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_3NegSum_3*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_3Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_3MeanNeg_3Const_3*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_2/tagsConst*
valueB Bloss_2*
dtype0*
_output_shapes
: 
M
loss_2ScalarSummaryloss_2/tagsMean_3*
T0*
_output_shapes
: 
T
gradients_2/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_2/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_2/FillFillgradients_2/Shapegradients_2/Const*
T0*
_output_shapes
: 
o
%gradients_2/Mean_3_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_2/Mean_3_grad/ReshapeReshapegradients_2/Fill%gradients_2/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_2/Mean_3_grad/ShapeShapeNeg_3*
T0*
out_type0*
_output_shapes
:
§
gradients_2/Mean_3_grad/TileTilegradients_2/Mean_3_grad/Reshapegradients_2/Mean_3_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
d
gradients_2/Mean_3_grad/Shape_1ShapeNeg_3*
out_type0*
_output_shapes
:*
T0
b
gradients_2/Mean_3_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_2/Mean_3_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_2/Mean_3_grad/ProdProdgradients_2/Mean_3_grad/Shape_1gradients_2/Mean_3_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_2/Mean_3_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
¶
gradients_2/Mean_3_grad/Prod_1Prodgradients_2/Mean_3_grad/Shape_2gradients_2/Mean_3_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_2/Mean_3_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_2/Mean_3_grad/MaximumMaximumgradients_2/Mean_3_grad/Prod_1!gradients_2/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_2/Mean_3_grad/floordivFloorDivgradients_2/Mean_3_grad/Prodgradients_2/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_2/Mean_3_grad/CastCast gradients_2/Mean_3_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_2/Mean_3_grad/truedivRealDivgradients_2/Mean_3_grad/Tilegradients_2/Mean_3_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_2/Neg_3_grad/NegNeggradients_2/Mean_3_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_2/Sum_3_grad/ShapeShapemul_3*
out_type0*
_output_shapes
:*
T0
]
gradients_2/Sum_3_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_2/Sum_3_grad/addAddSum_3/reduction_indicesgradients_2/Sum_3_grad/Size*
T0*
_output_shapes
:
Д
gradients_2/Sum_3_grad/modFloorModgradients_2/Sum_3_grad/addgradients_2/Sum_3_grad/Size*
T0*
_output_shapes
:
h
gradients_2/Sum_3_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
d
"gradients_2/Sum_3_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"gradients_2/Sum_3_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
≤
gradients_2/Sum_3_grad/rangeRange"gradients_2/Sum_3_grad/range/startgradients_2/Sum_3_grad/Size"gradients_2/Sum_3_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_2/Sum_3_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_2/Sum_3_grad/FillFillgradients_2/Sum_3_grad/Shape_1!gradients_2/Sum_3_grad/Fill/value*
T0*
_output_shapes
:
б
$gradients_2/Sum_3_grad/DynamicStitchDynamicStitchgradients_2/Sum_3_grad/rangegradients_2/Sum_3_grad/modgradients_2/Sum_3_grad/Shapegradients_2/Sum_3_grad/Fill*
T0*
N*#
_output_shapes
:€€€€€€€€€
b
 gradients_2/Sum_3_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_2/Sum_3_grad/MaximumMaximum$gradients_2/Sum_3_grad/DynamicStitch gradients_2/Sum_3_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_2/Sum_3_grad/floordivFloorDivgradients_2/Sum_3_grad/Shapegradients_2/Sum_3_grad/Maximum*
T0*
_output_shapes
:
Ь
gradients_2/Sum_3_grad/ReshapeReshapegradients_2/Neg_3_grad/Neg$gradients_2/Sum_3_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
®
gradients_2/Sum_3_grad/TileTilegradients_2/Sum_3_grad/Reshapegradients_2/Sum_3_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_2/mul_3_grad/ShapeShapePlaceholder_27*
out_type0*
_output_shapes
:*
T0
c
gradients_2/mul_3_grad/Shape_1ShapeLog_3*
out_type0*
_output_shapes
:*
T0
ј
,gradients_2/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/mul_3_grad/Shapegradients_2/mul_3_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
w
gradients_2/mul_3_grad/mulMulgradients_2/Sum_3_grad/TileLog_3*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients_2/mul_3_grad/SumSumgradients_2/mul_3_grad/mul,gradients_2/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients_2/mul_3_grad/ReshapeReshapegradients_2/mul_3_grad/Sumgradients_2/mul_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

В
gradients_2/mul_3_grad/mul_1MulPlaceholder_27gradients_2/Sum_3_grad/Tile*'
_output_shapes
:€€€€€€€€€
*
T0
±
gradients_2/mul_3_grad/Sum_1Sumgradients_2/mul_3_grad/mul_1.gradients_2/mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_2/mul_3_grad/Reshape_1Reshapegradients_2/mul_3_grad/Sum_1gradients_2/mul_3_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

s
'gradients_2/mul_3_grad/tuple/group_depsNoOp^gradients_2/mul_3_grad/Reshape!^gradients_2/mul_3_grad/Reshape_1
к
/gradients_2/mul_3_grad/tuple/control_dependencyIdentitygradients_2/mul_3_grad/Reshape(^gradients_2/mul_3_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_2/mul_3_grad/Reshape
р
1gradients_2/mul_3_grad/tuple/control_dependency_1Identity gradients_2/mul_3_grad/Reshape_1(^gradients_2/mul_3_grad/tuple/group_deps*3
_class)
'%loc:@gradients_2/mul_3_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
*
T0
†
!gradients_2/Log_3_grad/Reciprocal
Reciprocal	Softmax_32^gradients_2/mul_3_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_2/Log_3_grad/mulMul1gradients_2/mul_3_grad/tuple/control_dependency_1!gradients_2/Log_3_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

~
gradients_2/Softmax_3_grad/mulMulgradients_2/Log_3_grad/mul	Softmax_3*'
_output_shapes
:€€€€€€€€€
*
T0
z
0gradients_2/Softmax_3_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_2/Softmax_3_grad/SumSumgradients_2/Softmax_3_grad/mul0gradients_2/Softmax_3_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_2/Softmax_3_grad/Reshape/shapeConst*
valueB"€€€€   *
_output_shapes
:*
dtype0
Ј
"gradients_2/Softmax_3_grad/ReshapeReshapegradients_2/Softmax_3_grad/Sum(gradients_2/Softmax_3_grad/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Ч
gradients_2/Softmax_3_grad/subSubgradients_2/Log_3_grad/mul"gradients_2/Softmax_3_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_2/Softmax_3_grad/mul_1Mulgradients_2/Softmax_3_grad/sub	Softmax_3*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_2/add_33_grad/ShapeShape	MatMul_16*
_output_shapes
:*
out_type0*
T0
p
gradients_2/add_33_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   
   
√
-gradients_2/add_33_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_33_grad/Shapegradients_2/add_33_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_2/add_33_grad/SumSum gradients_2/Softmax_3_grad/mul_1-gradients_2/add_33_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_2/add_33_grad/ReshapeReshapegradients_2/add_33_grad/Sumgradients_2/add_33_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

Ј
gradients_2/add_33_grad/Sum_1Sum gradients_2/Softmax_3_grad/mul_1/gradients_2/add_33_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_2/add_33_grad/Reshape_1Reshapegradients_2/add_33_grad/Sum_1gradients_2/add_33_grad/Shape_1*
_output_shapes

:
*
Tshape0*
T0
v
(gradients_2/add_33_grad/tuple/group_depsNoOp ^gradients_2/add_33_grad/Reshape"^gradients_2/add_33_grad/Reshape_1
о
0gradients_2/add_33_grad/tuple/control_dependencyIdentitygradients_2/add_33_grad/Reshape)^gradients_2/add_33_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_2/add_33_grad/Reshape*'
_output_shapes
:€€€€€€€€€

л
2gradients_2/add_33_grad/tuple/control_dependency_1Identity!gradients_2/add_33_grad/Reshape_1)^gradients_2/add_33_grad/tuple/group_deps*
_output_shapes

:
*4
_class*
(&loc:@gradients_2/add_33_grad/Reshape_1*
T0
«
!gradients_2/MatMul_16_grad/MatMulMatMul0gradients_2/add_33_grad/tuple/control_dependencyVariable_32/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_2/MatMul_16_grad/MatMul_1MatMulTanh_100gradients_2/add_33_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_2/MatMul_16_grad/tuple/group_depsNoOp"^gradients_2/MatMul_16_grad/MatMul$^gradients_2/MatMul_16_grad/MatMul_1
ш
3gradients_2/MatMul_16_grad/tuple/control_dependencyIdentity!gradients_2/MatMul_16_grad/MatMul,^gradients_2/MatMul_16_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*4
_class*
(&loc:@gradients_2/MatMul_16_grad/MatMul
х
5gradients_2/MatMul_16_grad/tuple/control_dependency_1Identity#gradients_2/MatMul_16_grad/MatMul_1,^gradients_2/MatMul_16_grad/tuple/group_deps*
T0*
_output_shapes

:2
*6
_class,
*(loc:@gradients_2/MatMul_16_grad/MatMul_1
Э
!gradients_2/Tanh_10_grad/TanhGradTanhGradTanh_103gradients_2/MatMul_16_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
f
gradients_2/add_31_grad/ShapeShape	MatMul_15*
_output_shapes
:*
out_type0*
T0
p
gradients_2/add_31_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   2   
√
-gradients_2/add_31_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/add_31_grad/Shapegradients_2/add_31_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
gradients_2/add_31_grad/SumSum!gradients_2/Tanh_10_grad/TanhGrad-gradients_2/add_31_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_2/add_31_grad/ReshapeReshapegradients_2/add_31_grad/Sumgradients_2/add_31_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€2*
Tshape0
Є
gradients_2/add_31_grad/Sum_1Sum!gradients_2/Tanh_10_grad/TanhGrad/gradients_2/add_31_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_2/add_31_grad/Reshape_1Reshapegradients_2/add_31_grad/Sum_1gradients_2/add_31_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:2
v
(gradients_2/add_31_grad/tuple/group_depsNoOp ^gradients_2/add_31_grad/Reshape"^gradients_2/add_31_grad/Reshape_1
о
0gradients_2/add_31_grad/tuple/control_dependencyIdentitygradients_2/add_31_grad/Reshape)^gradients_2/add_31_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_2/add_31_grad/Reshape*'
_output_shapes
:€€€€€€€€€2
л
2gradients_2/add_31_grad/tuple/control_dependency_1Identity!gradients_2/add_31_grad/Reshape_1)^gradients_2/add_31_grad/tuple/group_deps*4
_class*
(&loc:@gradients_2/add_31_grad/Reshape_1*
_output_shapes

:2*
T0
«
!gradients_2/MatMul_15_grad/MatMulMatMul0gradients_2/add_31_grad/tuple/control_dependencyVariable_30/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
T0
Њ
#gradients_2/MatMul_15_grad/MatMul_1MatMulPlaceholder_260gradients_2/add_31_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_2/MatMul_15_grad/tuple/group_depsNoOp"^gradients_2/MatMul_15_grad/MatMul$^gradients_2/MatMul_15_grad/MatMul_1
ш
3gradients_2/MatMul_15_grad/tuple/control_dependencyIdentity!gradients_2/MatMul_15_grad/MatMul,^gradients_2/MatMul_15_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*4
_class*
(&loc:@gradients_2/MatMul_15_grad/MatMul*
T0
х
5gradients_2/MatMul_15_grad/tuple/control_dependency_1Identity#gradients_2/MatMul_15_grad/MatMul_1,^gradients_2/MatMul_15_grad/tuple/group_deps*
T0*
_output_shapes

:@2*6
_class,
*(loc:@gradients_2/MatMul_15_grad/MatMul_1
d
GradientDescent_2/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
Т
9GradientDescent_2/update_Variable_30/ApplyGradientDescentApplyGradientDescentVariable_30GradientDescent_2/learning_rate5gradients_2/MatMul_15_grad/tuple/control_dependency_1*
_output_shapes

:@2*
_class
loc:@Variable_30*
T0*
use_locking( 
П
9GradientDescent_2/update_Variable_31/ApplyGradientDescentApplyGradientDescentVariable_31GradientDescent_2/learning_rate2gradients_2/add_31_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_31*
_output_shapes

:2
Т
9GradientDescent_2/update_Variable_32/ApplyGradientDescentApplyGradientDescentVariable_32GradientDescent_2/learning_rate5gradients_2/MatMul_16_grad/tuple/control_dependency_1*
_class
loc:@Variable_32*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_2/update_Variable_33/ApplyGradientDescentApplyGradientDescentVariable_33GradientDescent_2/learning_rate2gradients_2/add_33_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_33*
_output_shapes

:

Й
GradientDescent_2NoOp:^GradientDescent_2/update_Variable_30/ApplyGradientDescent:^GradientDescent_2/update_Variable_31/ApplyGradientDescent:^GradientDescent_2/update_Variable_32/ApplyGradientDescent:^GradientDescent_2/update_Variable_33/ApplyGradientDescent
Q
Placeholder_28Placeholder*
_output_shapes
:*
shape: *
dtype0
s
Merge_1/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2*
N*
_output_shapes
: 
`
Placeholder_29Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_30Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_17/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Z
random_normal_17/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
random_normal_17/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_17/RandomStandardNormalRandomStandardNormalrandom_normal_17/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_17/mulMul%random_normal_17/RandomStandardNormalrandom_normal_17/stddev*
_output_shapes

:@2*
T0
m
random_normal_17Addrandom_normal_17/mulrandom_normal_17/mean*
T0*
_output_shapes

:@2

Variable_34
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_34/AssignAssignVariable_34random_normal_17*
_class
loc:@Variable_34*
_output_shapes

:@2*
T0*
validate_shape(*
use_locking(
r
Variable_34/readIdentityVariable_34*
_class
loc:@Variable_34*
_output_shapes

:@2*
T0
]
zeros_17Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_34/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
J
add_34Addzeros_17add_34/y*
_output_shapes

:2*
T0

Variable_35
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_35/AssignAssignVariable_35add_34*
use_locking(*
T0*
_class
loc:@Variable_35*
validate_shape(*
_output_shapes

:2
r
Variable_35/readIdentityVariable_35*
T0*
_output_shapes

:2*
_class
loc:@Variable_35
Н
	MatMul_17MatMulPlaceholder_29Variable_34/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_35Add	MatMul_17Variable_35/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_15/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
I
Tanh_11Tanhadd_35*
T0*'
_output_shapes
:€€€€€€€€€2
]
l1/outputs_1/tagConst*
valueB Bl1/outputs_1*
_output_shapes
: *
dtype0
\
l1/outputs_1HistogramSummaryl1/outputs_1/tagTanh_11*
T0*
_output_shapes
: 
g
random_normal_18/shapeConst*
valueB"2   
   *
_output_shapes
:*
dtype0
Z
random_normal_18/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_18/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
§
%random_normal_18/RandomStandardNormalRandomStandardNormalrandom_normal_18/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_18/mulMul%random_normal_18/RandomStandardNormalrandom_normal_18/stddev*
_output_shapes

:2
*
T0
m
random_normal_18Addrandom_normal_18/mulrandom_normal_18/mean*
_output_shapes

:2
*
T0

Variable_36
VariableV2*
_output_shapes

:2
*
	container *
dtype0*
shared_name *
shape
:2

≠
Variable_36/AssignAssignVariable_36random_normal_18*
_output_shapes

:2
*
validate_shape(*
_class
loc:@Variable_36*
T0*
use_locking(
r
Variable_36/readIdentityVariable_36*
_output_shapes

:2
*
_class
loc:@Variable_36*
T0
]
zeros_18Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_36/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_36Addzeros_18add_36/y*
T0*
_output_shapes

:


Variable_37
VariableV2*
_output_shapes

:
*
	container *
dtype0*
shared_name *
shape
:

£
Variable_37/AssignAssignVariable_37add_36*
_class
loc:@Variable_37*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
r
Variable_37/readIdentityVariable_37*
_class
loc:@Variable_37*
_output_shapes

:
*
T0
Ж
	MatMul_18MatMulTanh_11Variable_36/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
*
transpose_a( 
\
add_37Add	MatMul_18Variable_37/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_16/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
N
	Softmax_4Softmaxadd_37*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_1/tagConst*
valueB Bl2/outputs_1*
dtype0*
_output_shapes
: 
^
l2/outputs_1HistogramSummaryl2/outputs_1/tag	Softmax_4*
T0*
_output_shapes
: 
I
Log_4Log	Softmax_4*'
_output_shapes
:€€€€€€€€€
*
T0
U
mul_4MulPlaceholder_30Log_4*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_4/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
w
Sum_4Summul_4Sum_4/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
A
Neg_4NegSum_4*
T0*#
_output_shapes
:€€€€€€€€€
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_4MeanNeg_4Const_4*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_3/tagsConst*
valueB Bloss_3*
dtype0*
_output_shapes
: 
M
loss_3ScalarSummaryloss_3/tagsMean_4*
T0*
_output_shapes
: 
T
gradients_3/ShapeConst*
valueB *
_output_shapes
: *
dtype0
V
gradients_3/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_3/FillFillgradients_3/Shapegradients_3/Const*
_output_shapes
: *
T0
o
%gradients_3/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_3/Mean_4_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_4_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_3/Mean_4_grad/ShapeShapeNeg_4*
out_type0*
_output_shapes
:*
T0
§
gradients_3/Mean_4_grad/TileTilegradients_3/Mean_4_grad/Reshapegradients_3/Mean_4_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_3/Mean_4_grad/Shape_1ShapeNeg_4*
T0*
out_type0*
_output_shapes
:
b
gradients_3/Mean_4_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_3/Mean_4_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_3/Mean_4_grad/ProdProdgradients_3/Mean_4_grad/Shape_1gradients_3/Mean_4_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_3/Mean_4_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
¶
gradients_3/Mean_4_grad/Prod_1Prodgradients_3/Mean_4_grad/Shape_2gradients_3/Mean_4_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_3/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
О
gradients_3/Mean_4_grad/MaximumMaximumgradients_3/Mean_4_grad/Prod_1!gradients_3/Mean_4_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_3/Mean_4_grad/floordivFloorDivgradients_3/Mean_4_grad/Prodgradients_3/Mean_4_grad/Maximum*
_output_shapes
: *
T0
v
gradients_3/Mean_4_grad/CastCast gradients_3/Mean_4_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_3/Mean_4_grad/truedivRealDivgradients_3/Mean_4_grad/Tilegradients_3/Mean_4_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
p
gradients_3/Neg_4_grad/NegNeggradients_3/Mean_4_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
a
gradients_3/Sum_4_grad/ShapeShapemul_4*
out_type0*
_output_shapes
:*
T0
]
gradients_3/Sum_4_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_3/Sum_4_grad/addAddSum_4/reduction_indicesgradients_3/Sum_4_grad/Size*
_output_shapes
:*
T0
Д
gradients_3/Sum_4_grad/modFloorModgradients_3/Sum_4_grad/addgradients_3/Sum_4_grad/Size*
T0*
_output_shapes
:
h
gradients_3/Sum_4_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
d
"gradients_3/Sum_4_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"gradients_3/Sum_4_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
≤
gradients_3/Sum_4_grad/rangeRange"gradients_3/Sum_4_grad/range/startgradients_3/Sum_4_grad/Size"gradients_3/Sum_4_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_3/Sum_4_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_3/Sum_4_grad/FillFillgradients_3/Sum_4_grad/Shape_1!gradients_3/Sum_4_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_3/Sum_4_grad/DynamicStitchDynamicStitchgradients_3/Sum_4_grad/rangegradients_3/Sum_4_grad/modgradients_3/Sum_4_grad/Shapegradients_3/Sum_4_grad/Fill*
T0*
N*#
_output_shapes
:€€€€€€€€€
b
 gradients_3/Sum_4_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_3/Sum_4_grad/MaximumMaximum$gradients_3/Sum_4_grad/DynamicStitch gradients_3/Sum_4_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_3/Sum_4_grad/floordivFloorDivgradients_3/Sum_4_grad/Shapegradients_3/Sum_4_grad/Maximum*
T0*
_output_shapes
:
Ь
gradients_3/Sum_4_grad/ReshapeReshapegradients_3/Neg_4_grad/Neg$gradients_3/Sum_4_grad/DynamicStitch*
Tshape0*
_output_shapes
:*
T0
®
gradients_3/Sum_4_grad/TileTilegradients_3/Sum_4_grad/Reshapegradients_3/Sum_4_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_3/mul_4_grad/ShapeShapePlaceholder_30*
T0*
out_type0*
_output_shapes
:
c
gradients_3/mul_4_grad/Shape_1ShapeLog_4*
T0*
out_type0*
_output_shapes
:
ј
,gradients_3/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_4_grad/Shapegradients_3/mul_4_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_3/mul_4_grad/mulMulgradients_3/Sum_4_grad/TileLog_4*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients_3/mul_4_grad/SumSumgradients_3/mul_4_grad/mul,gradients_3/mul_4_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients_3/mul_4_grad/ReshapeReshapegradients_3/mul_4_grad/Sumgradients_3/mul_4_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
В
gradients_3/mul_4_grad/mul_1MulPlaceholder_30gradients_3/Sum_4_grad/Tile*'
_output_shapes
:€€€€€€€€€
*
T0
±
gradients_3/mul_4_grad/Sum_1Sumgradients_3/mul_4_grad/mul_1.gradients_3/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
©
 gradients_3/mul_4_grad/Reshape_1Reshapegradients_3/mul_4_grad/Sum_1gradients_3/mul_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

s
'gradients_3/mul_4_grad/tuple/group_depsNoOp^gradients_3/mul_4_grad/Reshape!^gradients_3/mul_4_grad/Reshape_1
к
/gradients_3/mul_4_grad/tuple/control_dependencyIdentitygradients_3/mul_4_grad/Reshape(^gradients_3/mul_4_grad/tuple/group_deps*1
_class'
%#loc:@gradients_3/mul_4_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
р
1gradients_3/mul_4_grad/tuple/control_dependency_1Identity gradients_3/mul_4_grad/Reshape_1(^gradients_3/mul_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_3/mul_4_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_3/Log_4_grad/Reciprocal
Reciprocal	Softmax_42^gradients_3/mul_4_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_3/Log_4_grad/mulMul1gradients_3/mul_4_grad/tuple/control_dependency_1!gradients_3/Log_4_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_3/Softmax_4_grad/mulMulgradients_3/Log_4_grad/mul	Softmax_4*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_3/Softmax_4_grad/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
¬
gradients_3/Softmax_4_grad/SumSumgradients_3/Softmax_4_grad/mul0gradients_3/Softmax_4_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_3/Softmax_4_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients_3/Softmax_4_grad/ReshapeReshapegradients_3/Softmax_4_grad/Sum(gradients_3/Softmax_4_grad/Reshape/shape*'
_output_shapes
:€€€€€€€€€*
Tshape0*
T0
Ч
gradients_3/Softmax_4_grad/subSubgradients_3/Log_4_grad/mul"gradients_3/Softmax_4_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_3/Softmax_4_grad/mul_1Mulgradients_3/Softmax_4_grad/sub	Softmax_4*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_3/add_37_grad/ShapeShape	MatMul_18*
T0*
_output_shapes
:*
out_type0
p
gradients_3/add_37_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
√
-gradients_3/add_37_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_37_grad/Shapegradients_3/add_37_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_3/add_37_grad/SumSum gradients_3/Softmax_4_grad/mul_1-gradients_3/add_37_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_3/add_37_grad/ReshapeReshapegradients_3/add_37_grad/Sumgradients_3/add_37_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
Ј
gradients_3/add_37_grad/Sum_1Sum gradients_3/Softmax_4_grad/mul_1/gradients_3/add_37_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_3/add_37_grad/Reshape_1Reshapegradients_3/add_37_grad/Sum_1gradients_3/add_37_grad/Shape_1*
_output_shapes

:
*
Tshape0*
T0
v
(gradients_3/add_37_grad/tuple/group_depsNoOp ^gradients_3/add_37_grad/Reshape"^gradients_3/add_37_grad/Reshape_1
о
0gradients_3/add_37_grad/tuple/control_dependencyIdentitygradients_3/add_37_grad/Reshape)^gradients_3/add_37_grad/tuple/group_deps*2
_class(
&$loc:@gradients_3/add_37_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_3/add_37_grad/tuple/control_dependency_1Identity!gradients_3/add_37_grad/Reshape_1)^gradients_3/add_37_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_3/add_37_grad/Reshape_1*
_output_shapes

:

«
!gradients_3/MatMul_18_grad/MatMulMatMul0gradients_3/add_37_grad/tuple/control_dependencyVariable_36/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_3/MatMul_18_grad/MatMul_1MatMulTanh_110gradients_3/add_37_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
}
+gradients_3/MatMul_18_grad/tuple/group_depsNoOp"^gradients_3/MatMul_18_grad/MatMul$^gradients_3/MatMul_18_grad/MatMul_1
ш
3gradients_3/MatMul_18_grad/tuple/control_dependencyIdentity!gradients_3/MatMul_18_grad/MatMul,^gradients_3/MatMul_18_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*4
_class*
(&loc:@gradients_3/MatMul_18_grad/MatMul
х
5gradients_3/MatMul_18_grad/tuple/control_dependency_1Identity#gradients_3/MatMul_18_grad/MatMul_1,^gradients_3/MatMul_18_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_3/MatMul_18_grad/MatMul_1*
_output_shapes

:2

Э
!gradients_3/Tanh_11_grad/TanhGradTanhGradTanh_113gradients_3/MatMul_18_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_3/add_35_grad/ShapeShape	MatMul_17*
_output_shapes
:*
out_type0*
T0
p
gradients_3/add_35_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   2   
√
-gradients_3/add_35_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/add_35_grad/Shapegradients_3/add_35_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
gradients_3/add_35_grad/SumSum!gradients_3/Tanh_11_grad/TanhGrad-gradients_3/add_35_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_3/add_35_grad/ReshapeReshapegradients_3/add_35_grad/Sumgradients_3/add_35_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€2*
Tshape0
Є
gradients_3/add_35_grad/Sum_1Sum!gradients_3/Tanh_11_grad/TanhGrad/gradients_3/add_35_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_3/add_35_grad/Reshape_1Reshapegradients_3/add_35_grad/Sum_1gradients_3/add_35_grad/Shape_1*
Tshape0*
_output_shapes

:2*
T0
v
(gradients_3/add_35_grad/tuple/group_depsNoOp ^gradients_3/add_35_grad/Reshape"^gradients_3/add_35_grad/Reshape_1
о
0gradients_3/add_35_grad/tuple/control_dependencyIdentitygradients_3/add_35_grad/Reshape)^gradients_3/add_35_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_3/add_35_grad/Reshape
л
2gradients_3/add_35_grad/tuple/control_dependency_1Identity!gradients_3/add_35_grad/Reshape_1)^gradients_3/add_35_grad/tuple/group_deps*
_output_shapes

:2*4
_class*
(&loc:@gradients_3/add_35_grad/Reshape_1*
T0
«
!gradients_3/MatMul_17_grad/MatMulMatMul0gradients_3/add_35_grad/tuple/control_dependencyVariable_34/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
T0
Њ
#gradients_3/MatMul_17_grad/MatMul_1MatMulPlaceholder_290gradients_3/add_35_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_3/MatMul_17_grad/tuple/group_depsNoOp"^gradients_3/MatMul_17_grad/MatMul$^gradients_3/MatMul_17_grad/MatMul_1
ш
3gradients_3/MatMul_17_grad/tuple/control_dependencyIdentity!gradients_3/MatMul_17_grad/MatMul,^gradients_3/MatMul_17_grad/tuple/group_deps*4
_class*
(&loc:@gradients_3/MatMul_17_grad/MatMul*'
_output_shapes
:€€€€€€€€€@*
T0
х
5gradients_3/MatMul_17_grad/tuple/control_dependency_1Identity#gradients_3/MatMul_17_grad/MatMul_1,^gradients_3/MatMul_17_grad/tuple/group_deps*6
_class,
*(loc:@gradients_3/MatMul_17_grad/MatMul_1*
_output_shapes

:@2*
T0
d
GradientDescent_3/learning_rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Т
9GradientDescent_3/update_Variable_34/ApplyGradientDescentApplyGradientDescentVariable_34GradientDescent_3/learning_rate5gradients_3/MatMul_17_grad/tuple/control_dependency_1*
_class
loc:@Variable_34*
_output_shapes

:@2*
T0*
use_locking( 
П
9GradientDescent_3/update_Variable_35/ApplyGradientDescentApplyGradientDescentVariable_35GradientDescent_3/learning_rate2gradients_3/add_35_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_35*
_output_shapes

:2
Т
9GradientDescent_3/update_Variable_36/ApplyGradientDescentApplyGradientDescentVariable_36GradientDescent_3/learning_rate5gradients_3/MatMul_18_grad/tuple/control_dependency_1*
_output_shapes

:2
*
_class
loc:@Variable_36*
T0*
use_locking( 
П
9GradientDescent_3/update_Variable_37/ApplyGradientDescentApplyGradientDescentVariable_37GradientDescent_3/learning_rate2gradients_3/add_37_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:
*
_class
loc:@Variable_37
Й
GradientDescent_3NoOp:^GradientDescent_3/update_Variable_34/ApplyGradientDescent:^GradientDescent_3/update_Variable_35/ApplyGradientDescent:^GradientDescent_3/update_Variable_36/ApplyGradientDescent:^GradientDescent_3/update_Variable_37/ApplyGradientDescent
Q
Placeholder_31Placeholder*
dtype0*
shape: *
_output_shapes
:
Ч
Merge_2/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3*
_output_shapes
: *
N
`
Placeholder_32Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_33Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_19/shapeConst*
valueB"@   2   *
dtype0*
_output_shapes
:
Z
random_normal_19/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_19/stddevConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
§
%random_normal_19/RandomStandardNormalRandomStandardNormalrandom_normal_19/shape*
_output_shapes

:@2*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_19/mulMul%random_normal_19/RandomStandardNormalrandom_normal_19/stddev*
_output_shapes

:@2*
T0
m
random_normal_19Addrandom_normal_19/mulrandom_normal_19/mean*
_output_shapes

:@2*
T0

Variable_38
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_38/AssignAssignVariable_38random_normal_19*
use_locking(*
T0*
_class
loc:@Variable_38*
validate_shape(*
_output_shapes

:@2
r
Variable_38/readIdentityVariable_38*
_class
loc:@Variable_38*
_output_shapes

:@2*
T0
]
zeros_19Const*
_output_shapes

:2*
dtype0*
valueB2*    
M
add_38/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_38Addzeros_19add_38/y*
_output_shapes

:2*
T0

Variable_39
VariableV2*
shared_name *
dtype0*
shape
:2*
_output_shapes

:2*
	container 
£
Variable_39/AssignAssignVariable_39add_38*
use_locking(*
T0*
_class
loc:@Variable_39*
validate_shape(*
_output_shapes

:2
r
Variable_39/readIdentityVariable_39*
_output_shapes

:2*
_class
loc:@Variable_39*
T0
Н
	MatMul_19MatMulPlaceholder_32Variable_38/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_39Add	MatMul_19Variable_39/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_17/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
I
Tanh_12Tanhadd_39*
T0*'
_output_shapes
:€€€€€€€€€2
]
l1/outputs_2/tagConst*
valueB Bl1/outputs_2*
_output_shapes
: *
dtype0
\
l1/outputs_2HistogramSummaryl1/outputs_2/tagTanh_12*
T0*
_output_shapes
: 
g
random_normal_20/shapeConst*
valueB"2   
   *
dtype0*
_output_shapes
:
Z
random_normal_20/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_20/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_20/RandomStandardNormalRandomStandardNormalrandom_normal_20/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_20/mulMul%random_normal_20/RandomStandardNormalrandom_normal_20/stddev*
_output_shapes

:2
*
T0
m
random_normal_20Addrandom_normal_20/mulrandom_normal_20/mean*
T0*
_output_shapes

:2


Variable_40
VariableV2*
shared_name *
dtype0*
shape
:2
*
_output_shapes

:2
*
	container 
≠
Variable_40/AssignAssignVariable_40random_normal_20*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2
*
_class
loc:@Variable_40
r
Variable_40/readIdentityVariable_40*
_class
loc:@Variable_40*
_output_shapes

:2
*
T0
]
zeros_20Const*
dtype0*
_output_shapes

:
*
valueB
*    
M
add_40/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
J
add_40Addzeros_20add_40/y*
_output_shapes

:
*
T0

Variable_41
VariableV2*
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*
	container 
£
Variable_41/AssignAssignVariable_41add_40*
_class
loc:@Variable_41*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
r
Variable_41/readIdentityVariable_41*
T0*
_class
loc:@Variable_41*
_output_shapes

:

Ж
	MatMul_20MatMulTanh_12Variable_40/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_41Add	MatMul_20Variable_41/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_18/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_5Softmaxadd_41*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_2/tagConst*
valueB Bl2/outputs_2*
dtype0*
_output_shapes
: 
^
l2/outputs_2HistogramSummaryl2/outputs_2/tag	Softmax_5*
_output_shapes
: *
T0
I
Log_5Log	Softmax_5*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_5MulPlaceholder_33Log_5*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_5/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
w
Sum_5Summul_5Sum_5/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_5NegSum_5*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_5Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_5MeanNeg_5Const_5*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
loss_4/tagsConst*
dtype0*
_output_shapes
: *
valueB Bloss_4
M
loss_4ScalarSummaryloss_4/tagsMean_5*
T0*
_output_shapes
: 
T
gradients_4/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
_
gradients_4/FillFillgradients_4/Shapegradients_4/Const*
T0*
_output_shapes
: 
o
%gradients_4/Mean_5_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ц
gradients_4/Mean_5_grad/ReshapeReshapegradients_4/Fill%gradients_4/Mean_5_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients_4/Mean_5_grad/ShapeShapeNeg_5*
_output_shapes
:*
out_type0*
T0
§
gradients_4/Mean_5_grad/TileTilegradients_4/Mean_5_grad/Reshapegradients_4/Mean_5_grad/Shape*#
_output_shapes
:€€€€€€€€€*
T0*

Tmultiples0
d
gradients_4/Mean_5_grad/Shape_1ShapeNeg_5*
out_type0*
_output_shapes
:*
T0
b
gradients_4/Mean_5_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_4/Mean_5_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_4/Mean_5_grad/ProdProdgradients_4/Mean_5_grad/Shape_1gradients_4/Mean_5_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_4/Mean_5_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
¶
gradients_4/Mean_5_grad/Prod_1Prodgradients_4/Mean_5_grad/Shape_2gradients_4/Mean_5_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_4/Mean_5_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_4/Mean_5_grad/MaximumMaximumgradients_4/Mean_5_grad/Prod_1!gradients_4/Mean_5_grad/Maximum/y*
_output_shapes
: *
T0
М
 gradients_4/Mean_5_grad/floordivFloorDivgradients_4/Mean_5_grad/Prodgradients_4/Mean_5_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_4/Mean_5_grad/CastCast gradients_4/Mean_5_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_4/Mean_5_grad/truedivRealDivgradients_4/Mean_5_grad/Tilegradients_4/Mean_5_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_4/Neg_5_grad/NegNeggradients_4/Mean_5_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_4/Sum_5_grad/ShapeShapemul_5*
_output_shapes
:*
out_type0*
T0
]
gradients_4/Sum_5_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
gradients_4/Sum_5_grad/addAddSum_5/reduction_indicesgradients_4/Sum_5_grad/Size*
T0*
_output_shapes
:
Д
gradients_4/Sum_5_grad/modFloorModgradients_4/Sum_5_grad/addgradients_4/Sum_5_grad/Size*
T0*
_output_shapes
:
h
gradients_4/Sum_5_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
d
"gradients_4/Sum_5_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0
d
"gradients_4/Sum_5_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
≤
gradients_4/Sum_5_grad/rangeRange"gradients_4/Sum_5_grad/range/startgradients_4/Sum_5_grad/Size"gradients_4/Sum_5_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_4/Sum_5_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_4/Sum_5_grad/FillFillgradients_4/Sum_5_grad/Shape_1!gradients_4/Sum_5_grad/Fill/value*
T0*
_output_shapes
:
б
$gradients_4/Sum_5_grad/DynamicStitchDynamicStitchgradients_4/Sum_5_grad/rangegradients_4/Sum_5_grad/modgradients_4/Sum_5_grad/Shapegradients_4/Sum_5_grad/Fill*
T0*
N*#
_output_shapes
:€€€€€€€€€
b
 gradients_4/Sum_5_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
Я
gradients_4/Sum_5_grad/MaximumMaximum$gradients_4/Sum_5_grad/DynamicStitch gradients_4/Sum_5_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
О
gradients_4/Sum_5_grad/floordivFloorDivgradients_4/Sum_5_grad/Shapegradients_4/Sum_5_grad/Maximum*
T0*
_output_shapes
:
Ь
gradients_4/Sum_5_grad/ReshapeReshapegradients_4/Neg_5_grad/Neg$gradients_4/Sum_5_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
®
gradients_4/Sum_5_grad/TileTilegradients_4/Sum_5_grad/Reshapegradients_4/Sum_5_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_4/mul_5_grad/ShapeShapePlaceholder_33*
_output_shapes
:*
out_type0*
T0
c
gradients_4/mul_5_grad/Shape_1ShapeLog_5*
out_type0*
_output_shapes
:*
T0
ј
,gradients_4/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/mul_5_grad/Shapegradients_4/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
w
gradients_4/mul_5_grad/mulMulgradients_4/Sum_5_grad/TileLog_5*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_4/mul_5_grad/SumSumgradients_4/mul_5_grad/mul,gradients_4/mul_5_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients_4/mul_5_grad/ReshapeReshapegradients_4/mul_5_grad/Sumgradients_4/mul_5_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
В
gradients_4/mul_5_grad/mul_1MulPlaceholder_33gradients_4/Sum_5_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_4/mul_5_grad/Sum_1Sumgradients_4/mul_5_grad/mul_1.gradients_4/mul_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_4/mul_5_grad/Reshape_1Reshapegradients_4/mul_5_grad/Sum_1gradients_4/mul_5_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
s
'gradients_4/mul_5_grad/tuple/group_depsNoOp^gradients_4/mul_5_grad/Reshape!^gradients_4/mul_5_grad/Reshape_1
к
/gradients_4/mul_5_grad/tuple/control_dependencyIdentitygradients_4/mul_5_grad/Reshape(^gradients_4/mul_5_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_4/mul_5_grad/Reshape
р
1gradients_4/mul_5_grad/tuple/control_dependency_1Identity gradients_4/mul_5_grad/Reshape_1(^gradients_4/mul_5_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*3
_class)
'%loc:@gradients_4/mul_5_grad/Reshape_1*
T0
†
!gradients_4/Log_5_grad/Reciprocal
Reciprocal	Softmax_52^gradients_4/mul_5_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_4/Log_5_grad/mulMul1gradients_4/mul_5_grad/tuple/control_dependency_1!gradients_4/Log_5_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

~
gradients_4/Softmax_5_grad/mulMulgradients_4/Log_5_grad/mul	Softmax_5*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_4/Softmax_5_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_4/Softmax_5_grad/SumSumgradients_4/Softmax_5_grad/mul0gradients_4/Softmax_5_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_4/Softmax_5_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
Ј
"gradients_4/Softmax_5_grad/ReshapeReshapegradients_4/Softmax_5_grad/Sum(gradients_4/Softmax_5_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ч
gradients_4/Softmax_5_grad/subSubgradients_4/Log_5_grad/mul"gradients_4/Softmax_5_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_4/Softmax_5_grad/mul_1Mulgradients_4/Softmax_5_grad/sub	Softmax_5*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_4/add_41_grad/ShapeShape	MatMul_20*
_output_shapes
:*
out_type0*
T0
p
gradients_4/add_41_grad/Shape_1Const*
valueB"   
   *
dtype0*
_output_shapes
:
√
-gradients_4/add_41_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_41_grad/Shapegradients_4/add_41_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
≥
gradients_4/add_41_grad/SumSum gradients_4/Softmax_5_grad/mul_1-gradients_4/add_41_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_4/add_41_grad/ReshapeReshapegradients_4/add_41_grad/Sumgradients_4/add_41_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

Ј
gradients_4/add_41_grad/Sum_1Sum gradients_4/Softmax_5_grad/mul_1/gradients_4/add_41_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_4/add_41_grad/Reshape_1Reshapegradients_4/add_41_grad/Sum_1gradients_4/add_41_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
v
(gradients_4/add_41_grad/tuple/group_depsNoOp ^gradients_4/add_41_grad/Reshape"^gradients_4/add_41_grad/Reshape_1
о
0gradients_4/add_41_grad/tuple/control_dependencyIdentitygradients_4/add_41_grad/Reshape)^gradients_4/add_41_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*2
_class(
&$loc:@gradients_4/add_41_grad/Reshape*
T0
л
2gradients_4/add_41_grad/tuple/control_dependency_1Identity!gradients_4/add_41_grad/Reshape_1)^gradients_4/add_41_grad/tuple/group_deps*
T0*
_output_shapes

:
*4
_class*
(&loc:@gradients_4/add_41_grad/Reshape_1
«
!gradients_4/MatMul_20_grad/MatMulMatMul0gradients_4/add_41_grad/tuple/control_dependencyVariable_40/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_4/MatMul_20_grad/MatMul_1MatMulTanh_120gradients_4/add_41_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_4/MatMul_20_grad/tuple/group_depsNoOp"^gradients_4/MatMul_20_grad/MatMul$^gradients_4/MatMul_20_grad/MatMul_1
ш
3gradients_4/MatMul_20_grad/tuple/control_dependencyIdentity!gradients_4/MatMul_20_grad/MatMul,^gradients_4/MatMul_20_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*4
_class*
(&loc:@gradients_4/MatMul_20_grad/MatMul*
T0
х
5gradients_4/MatMul_20_grad/tuple/control_dependency_1Identity#gradients_4/MatMul_20_grad/MatMul_1,^gradients_4/MatMul_20_grad/tuple/group_deps*
T0*
_output_shapes

:2
*6
_class,
*(loc:@gradients_4/MatMul_20_grad/MatMul_1
Э
!gradients_4/Tanh_12_grad/TanhGradTanhGradTanh_123gradients_4/MatMul_20_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
f
gradients_4/add_39_grad/ShapeShape	MatMul_19*
T0*
_output_shapes
:*
out_type0
p
gradients_4/add_39_grad/Shape_1Const*
valueB"   2   *
_output_shapes
:*
dtype0
√
-gradients_4/add_39_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/add_39_grad/Shapegradients_4/add_39_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
і
gradients_4/add_39_grad/SumSum!gradients_4/Tanh_12_grad/TanhGrad-gradients_4/add_39_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_4/add_39_grad/ReshapeReshapegradients_4/add_39_grad/Sumgradients_4/add_39_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€2
Є
gradients_4/add_39_grad/Sum_1Sum!gradients_4/Tanh_12_grad/TanhGrad/gradients_4/add_39_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_4/add_39_grad/Reshape_1Reshapegradients_4/add_39_grad/Sum_1gradients_4/add_39_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:2
v
(gradients_4/add_39_grad/tuple/group_depsNoOp ^gradients_4/add_39_grad/Reshape"^gradients_4/add_39_grad/Reshape_1
о
0gradients_4/add_39_grad/tuple/control_dependencyIdentitygradients_4/add_39_grad/Reshape)^gradients_4/add_39_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_4/add_39_grad/Reshape*
T0
л
2gradients_4/add_39_grad/tuple/control_dependency_1Identity!gradients_4/add_39_grad/Reshape_1)^gradients_4/add_39_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_4/add_39_grad/Reshape_1*
_output_shapes

:2
«
!gradients_4/MatMul_19_grad/MatMulMatMul0gradients_4/add_39_grad/tuple/control_dependencyVariable_38/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_4/MatMul_19_grad/MatMul_1MatMulPlaceholder_320gradients_4/add_39_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_4/MatMul_19_grad/tuple/group_depsNoOp"^gradients_4/MatMul_19_grad/MatMul$^gradients_4/MatMul_19_grad/MatMul_1
ш
3gradients_4/MatMul_19_grad/tuple/control_dependencyIdentity!gradients_4/MatMul_19_grad/MatMul,^gradients_4/MatMul_19_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_4/MatMul_19_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
х
5gradients_4/MatMul_19_grad/tuple/control_dependency_1Identity#gradients_4/MatMul_19_grad/MatMul_1,^gradients_4/MatMul_19_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_4/MatMul_19_grad/MatMul_1*
T0
d
GradientDescent_4/learning_rateConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
Т
9GradientDescent_4/update_Variable_38/ApplyGradientDescentApplyGradientDescentVariable_38GradientDescent_4/learning_rate5gradients_4/MatMul_19_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_38*
_output_shapes

:@2
П
9GradientDescent_4/update_Variable_39/ApplyGradientDescentApplyGradientDescentVariable_39GradientDescent_4/learning_rate2gradients_4/add_39_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_39*
_output_shapes

:2
Т
9GradientDescent_4/update_Variable_40/ApplyGradientDescentApplyGradientDescentVariable_40GradientDescent_4/learning_rate5gradients_4/MatMul_20_grad/tuple/control_dependency_1*
_class
loc:@Variable_40*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_4/update_Variable_41/ApplyGradientDescentApplyGradientDescentVariable_41GradientDescent_4/learning_rate2gradients_4/add_41_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_41*
_output_shapes

:

Й
GradientDescent_4NoOp:^GradientDescent_4/update_Variable_38/ApplyGradientDescent:^GradientDescent_4/update_Variable_39/ApplyGradientDescent:^GradientDescent_4/update_Variable_40/ApplyGradientDescent:^GradientDescent_4/update_Variable_41/ApplyGradientDescent
Q
Placeholder_34Placeholder*
_output_shapes
:*
dtype0*
shape: 
ї
Merge_3/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4*
N*
_output_shapes
: 
т
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign
`
Placeholder_35Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_36Placeholder*'
_output_shapes
:€€€€€€€€€
*
shape: *
dtype0
g
random_normal_21/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Z
random_normal_21/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_21/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_21/RandomStandardNormalRandomStandardNormalrandom_normal_21/shape*

seed *
T0*
dtype0*
_output_shapes

:@2*
seed2 
Д
random_normal_21/mulMul%random_normal_21/RandomStandardNormalrandom_normal_21/stddev*
T0*
_output_shapes

:@2
m
random_normal_21Addrandom_normal_21/mulrandom_normal_21/mean*
_output_shapes

:@2*
T0

Variable_42
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_42/AssignAssignVariable_42random_normal_21*
_output_shapes

:@2*
validate_shape(*
_class
loc:@Variable_42*
T0*
use_locking(
r
Variable_42/readIdentityVariable_42*
_class
loc:@Variable_42*
_output_shapes

:@2*
T0
]
zeros_21Const*
valueB2*    *
_output_shapes

:2*
dtype0
M
add_42/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ќћћ=
J
add_42Addzeros_21add_42/y*
T0*
_output_shapes

:2

Variable_43
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_43/AssignAssignVariable_43add_42*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_43
r
Variable_43/readIdentityVariable_43*
_output_shapes

:2*
_class
loc:@Variable_43*
T0
Н
	MatMul_21MatMulPlaceholder_35Variable_42/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_43Add	MatMul_21Variable_43/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_19/keep_probConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
I
Tanh_13Tanhadd_43*'
_output_shapes
:€€€€€€€€€2*
T0
]
l1/outputs_3/tagConst*
valueB Bl1/outputs_3*
dtype0*
_output_shapes
: 
\
l1/outputs_3HistogramSummaryl1/outputs_3/tagTanh_13*
_output_shapes
: *
T0
g
random_normal_22/shapeConst*
dtype0*
_output_shapes
:*
valueB"2   
   
Z
random_normal_22/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
random_normal_22/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_22/RandomStandardNormalRandomStandardNormalrandom_normal_22/shape*
dtype0*

seed *
T0*
_output_shapes

:2
*
seed2 
Д
random_normal_22/mulMul%random_normal_22/RandomStandardNormalrandom_normal_22/stddev*
_output_shapes

:2
*
T0
m
random_normal_22Addrandom_normal_22/mulrandom_normal_22/mean*
_output_shapes

:2
*
T0

Variable_44
VariableV2*
shape
:2
*
shared_name *
dtype0*
_output_shapes

:2
*
	container 
≠
Variable_44/AssignAssignVariable_44random_normal_22*
use_locking(*
T0*
_class
loc:@Variable_44*
validate_shape(*
_output_shapes

:2

r
Variable_44/readIdentityVariable_44*
T0*
_output_shapes

:2
*
_class
loc:@Variable_44
]
zeros_22Const*
valueB
*    *
_output_shapes

:
*
dtype0
M
add_44/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_44Addzeros_22add_44/y*
_output_shapes

:
*
T0

Variable_45
VariableV2*
_output_shapes

:
*
	container *
dtype0*
shared_name *
shape
:

£
Variable_45/AssignAssignVariable_45add_44*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
*
_class
loc:@Variable_45
r
Variable_45/readIdentityVariable_45*
_output_shapes

:
*
_class
loc:@Variable_45*
T0
Ж
	MatMul_22MatMulTanh_13Variable_44/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_45Add	MatMul_22Variable_45/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_20/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_6Softmaxadd_45*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_3/tagConst*
valueB Bl2/outputs_3*
dtype0*
_output_shapes
: 
^
l2/outputs_3HistogramSummaryl2/outputs_3/tag	Softmax_6*
_output_shapes
: *
T0
I
Log_6Log	Softmax_6*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_6MulPlaceholder_36Log_6*
T0*'
_output_shapes
:€€€€€€€€€

a
Sum_6/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
w
Sum_6Summul_6Sum_6/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_6NegSum_6*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_6Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_6MeanNeg_6Const_6*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
loss_5/tagsConst*
valueB Bloss_5*
_output_shapes
: *
dtype0
M
loss_5ScalarSummaryloss_5/tagsMean_6*
T0*
_output_shapes
: 
T
gradients_5/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
V
gradients_5/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
gradients_5/FillFillgradients_5/Shapegradients_5/Const*
_output_shapes
: *
T0
o
%gradients_5/Mean_6_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
Ц
gradients_5/Mean_6_grad/ReshapeReshapegradients_5/Fill%gradients_5/Mean_6_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_5/Mean_6_grad/ShapeShapeNeg_6*
T0*
out_type0*
_output_shapes
:
§
gradients_5/Mean_6_grad/TileTilegradients_5/Mean_6_grad/Reshapegradients_5/Mean_6_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
d
gradients_5/Mean_6_grad/Shape_1ShapeNeg_6*
out_type0*
_output_shapes
:*
T0
b
gradients_5/Mean_6_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
g
gradients_5/Mean_6_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_5/Mean_6_grad/ProdProdgradients_5/Mean_6_grad/Shape_1gradients_5/Mean_6_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_5/Mean_6_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
¶
gradients_5/Mean_6_grad/Prod_1Prodgradients_5/Mean_6_grad/Shape_2gradients_5/Mean_6_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_5/Mean_6_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_5/Mean_6_grad/MaximumMaximumgradients_5/Mean_6_grad/Prod_1!gradients_5/Mean_6_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_5/Mean_6_grad/floordivFloorDivgradients_5/Mean_6_grad/Prodgradients_5/Mean_6_grad/Maximum*
T0*
_output_shapes
: 
v
gradients_5/Mean_6_grad/CastCast gradients_5/Mean_6_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_5/Mean_6_grad/truedivRealDivgradients_5/Mean_6_grad/Tilegradients_5/Mean_6_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
p
gradients_5/Neg_6_grad/NegNeggradients_5/Mean_6_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_5/Sum_6_grad/ShapeShapemul_6*
T0*
_output_shapes
:*
out_type0
]
gradients_5/Sum_6_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
|
gradients_5/Sum_6_grad/addAddSum_6/reduction_indicesgradients_5/Sum_6_grad/Size*
T0*
_output_shapes
:
Д
gradients_5/Sum_6_grad/modFloorModgradients_5/Sum_6_grad/addgradients_5/Sum_6_grad/Size*
_output_shapes
:*
T0
h
gradients_5/Sum_6_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
d
"gradients_5/Sum_6_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"gradients_5/Sum_6_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
≤
gradients_5/Sum_6_grad/rangeRange"gradients_5/Sum_6_grad/range/startgradients_5/Sum_6_grad/Size"gradients_5/Sum_6_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_5/Sum_6_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_5/Sum_6_grad/FillFillgradients_5/Sum_6_grad/Shape_1!gradients_5/Sum_6_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_5/Sum_6_grad/DynamicStitchDynamicStitchgradients_5/Sum_6_grad/rangegradients_5/Sum_6_grad/modgradients_5/Sum_6_grad/Shapegradients_5/Sum_6_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
b
 gradients_5/Sum_6_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_5/Sum_6_grad/MaximumMaximum$gradients_5/Sum_6_grad/DynamicStitch gradients_5/Sum_6_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
О
gradients_5/Sum_6_grad/floordivFloorDivgradients_5/Sum_6_grad/Shapegradients_5/Sum_6_grad/Maximum*
_output_shapes
:*
T0
Ь
gradients_5/Sum_6_grad/ReshapeReshapegradients_5/Neg_6_grad/Neg$gradients_5/Sum_6_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
®
gradients_5/Sum_6_grad/TileTilegradients_5/Sum_6_grad/Reshapegradients_5/Sum_6_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_5/mul_6_grad/ShapeShapePlaceholder_36*
_output_shapes
:*
out_type0*
T0
c
gradients_5/mul_6_grad/Shape_1ShapeLog_6*
T0*
out_type0*
_output_shapes
:
ј
,gradients_5/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/mul_6_grad/Shapegradients_5/mul_6_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_5/mul_6_grad/mulMulgradients_5/Sum_6_grad/TileLog_6*
T0*'
_output_shapes
:€€€€€€€€€

Ђ
gradients_5/mul_6_grad/SumSumgradients_5/mul_6_grad/mul,gradients_5/mul_6_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
gradients_5/mul_6_grad/ReshapeReshapegradients_5/mul_6_grad/Sumgradients_5/mul_6_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
В
gradients_5/mul_6_grad/mul_1MulPlaceholder_36gradients_5/Sum_6_grad/Tile*'
_output_shapes
:€€€€€€€€€
*
T0
±
gradients_5/mul_6_grad/Sum_1Sumgradients_5/mul_6_grad/mul_1.gradients_5/mul_6_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
©
 gradients_5/mul_6_grad/Reshape_1Reshapegradients_5/mul_6_grad/Sum_1gradients_5/mul_6_grad/Shape_1*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
s
'gradients_5/mul_6_grad/tuple/group_depsNoOp^gradients_5/mul_6_grad/Reshape!^gradients_5/mul_6_grad/Reshape_1
к
/gradients_5/mul_6_grad/tuple/control_dependencyIdentitygradients_5/mul_6_grad/Reshape(^gradients_5/mul_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_5/mul_6_grad/Reshape*'
_output_shapes
:€€€€€€€€€

р
1gradients_5/mul_6_grad/tuple/control_dependency_1Identity gradients_5/mul_6_grad/Reshape_1(^gradients_5/mul_6_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_5/mul_6_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_5/Log_6_grad/Reciprocal
Reciprocal	Softmax_62^gradients_5/mul_6_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

©
gradients_5/Log_6_grad/mulMul1gradients_5/mul_6_grad/tuple/control_dependency_1!gradients_5/Log_6_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_5/Softmax_6_grad/mulMulgradients_5/Log_6_grad/mul	Softmax_6*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_5/Softmax_6_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
¬
gradients_5/Softmax_6_grad/SumSumgradients_5/Softmax_6_grad/mul0gradients_5/Softmax_6_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_5/Softmax_6_grad/Reshape/shapeConst*
valueB"€€€€   *
dtype0*
_output_shapes
:
Ј
"gradients_5/Softmax_6_grad/ReshapeReshapegradients_5/Softmax_6_grad/Sum(gradients_5/Softmax_6_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ч
gradients_5/Softmax_6_grad/subSubgradients_5/Log_6_grad/mul"gradients_5/Softmax_6_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_5/Softmax_6_grad/mul_1Mulgradients_5/Softmax_6_grad/sub	Softmax_6*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_5/add_45_grad/ShapeShape	MatMul_22*
_output_shapes
:*
out_type0*
T0
p
gradients_5/add_45_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
√
-gradients_5/add_45_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_45_grad/Shapegradients_5/add_45_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_5/add_45_grad/SumSum gradients_5/Softmax_6_grad/mul_1-gradients_5/add_45_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_5/add_45_grad/ReshapeReshapegradients_5/add_45_grad/Sumgradients_5/add_45_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€

Ј
gradients_5/add_45_grad/Sum_1Sum gradients_5/Softmax_6_grad/mul_1/gradients_5/add_45_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_5/add_45_grad/Reshape_1Reshapegradients_5/add_45_grad/Sum_1gradients_5/add_45_grad/Shape_1*
Tshape0*
_output_shapes

:
*
T0
v
(gradients_5/add_45_grad/tuple/group_depsNoOp ^gradients_5/add_45_grad/Reshape"^gradients_5/add_45_grad/Reshape_1
о
0gradients_5/add_45_grad/tuple/control_dependencyIdentitygradients_5/add_45_grad/Reshape)^gradients_5/add_45_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*2
_class(
&$loc:@gradients_5/add_45_grad/Reshape
л
2gradients_5/add_45_grad/tuple/control_dependency_1Identity!gradients_5/add_45_grad/Reshape_1)^gradients_5/add_45_grad/tuple/group_deps*
T0*
_output_shapes

:
*4
_class*
(&loc:@gradients_5/add_45_grad/Reshape_1
«
!gradients_5/MatMul_22_grad/MatMulMatMul0gradients_5/add_45_grad/tuple/control_dependencyVariable_44/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_5/MatMul_22_grad/MatMul_1MatMulTanh_130gradients_5/add_45_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_5/MatMul_22_grad/tuple/group_depsNoOp"^gradients_5/MatMul_22_grad/MatMul$^gradients_5/MatMul_22_grad/MatMul_1
ш
3gradients_5/MatMul_22_grad/tuple/control_dependencyIdentity!gradients_5/MatMul_22_grad/MatMul,^gradients_5/MatMul_22_grad/tuple/group_deps*4
_class*
(&loc:@gradients_5/MatMul_22_grad/MatMul*'
_output_shapes
:€€€€€€€€€2*
T0
х
5gradients_5/MatMul_22_grad/tuple/control_dependency_1Identity#gradients_5/MatMul_22_grad/MatMul_1,^gradients_5/MatMul_22_grad/tuple/group_deps*
T0*
_output_shapes

:2
*6
_class,
*(loc:@gradients_5/MatMul_22_grad/MatMul_1
Э
!gradients_5/Tanh_13_grad/TanhGradTanhGradTanh_133gradients_5/MatMul_22_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_5/add_43_grad/ShapeShape	MatMul_21*
_output_shapes
:*
out_type0*
T0
p
gradients_5/add_43_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   2   
√
-gradients_5/add_43_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_5/add_43_grad/Shapegradients_5/add_43_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
gradients_5/add_43_grad/SumSum!gradients_5/Tanh_13_grad/TanhGrad-gradients_5/add_43_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_5/add_43_grad/ReshapeReshapegradients_5/add_43_grad/Sumgradients_5/add_43_grad/Shape*'
_output_shapes
:€€€€€€€€€2*
Tshape0*
T0
Є
gradients_5/add_43_grad/Sum_1Sum!gradients_5/Tanh_13_grad/TanhGrad/gradients_5/add_43_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_5/add_43_grad/Reshape_1Reshapegradients_5/add_43_grad/Sum_1gradients_5/add_43_grad/Shape_1*
T0*
_output_shapes

:2*
Tshape0
v
(gradients_5/add_43_grad/tuple/group_depsNoOp ^gradients_5/add_43_grad/Reshape"^gradients_5/add_43_grad/Reshape_1
о
0gradients_5/add_43_grad/tuple/control_dependencyIdentitygradients_5/add_43_grad/Reshape)^gradients_5/add_43_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_5/add_43_grad/Reshape
л
2gradients_5/add_43_grad/tuple/control_dependency_1Identity!gradients_5/add_43_grad/Reshape_1)^gradients_5/add_43_grad/tuple/group_deps*
T0*
_output_shapes

:2*4
_class*
(&loc:@gradients_5/add_43_grad/Reshape_1
«
!gradients_5/MatMul_21_grad/MatMulMatMul0gradients_5/add_43_grad/tuple/control_dependencyVariable_42/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_5/MatMul_21_grad/MatMul_1MatMulPlaceholder_350gradients_5/add_43_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_5/MatMul_21_grad/tuple/group_depsNoOp"^gradients_5/MatMul_21_grad/MatMul$^gradients_5/MatMul_21_grad/MatMul_1
ш
3gradients_5/MatMul_21_grad/tuple/control_dependencyIdentity!gradients_5/MatMul_21_grad/MatMul,^gradients_5/MatMul_21_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*4
_class*
(&loc:@gradients_5/MatMul_21_grad/MatMul*
T0
х
5gradients_5/MatMul_21_grad/tuple/control_dependency_1Identity#gradients_5/MatMul_21_grad/MatMul_1,^gradients_5/MatMul_21_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_5/MatMul_21_grad/MatMul_1*
_output_shapes

:@2
d
GradientDescent_5/learning_rateConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
Т
9GradientDescent_5/update_Variable_42/ApplyGradientDescentApplyGradientDescentVariable_42GradientDescent_5/learning_rate5gradients_5/MatMul_21_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:@2*
_class
loc:@Variable_42
П
9GradientDescent_5/update_Variable_43/ApplyGradientDescentApplyGradientDescentVariable_43GradientDescent_5/learning_rate2gradients_5/add_43_grad/tuple/control_dependency_1*
_output_shapes

:2*
_class
loc:@Variable_43*
T0*
use_locking( 
Т
9GradientDescent_5/update_Variable_44/ApplyGradientDescentApplyGradientDescentVariable_44GradientDescent_5/learning_rate5gradients_5/MatMul_22_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:2
*
_class
loc:@Variable_44
П
9GradientDescent_5/update_Variable_45/ApplyGradientDescentApplyGradientDescentVariable_45GradientDescent_5/learning_rate2gradients_5/add_45_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_45*
_output_shapes

:

Й
GradientDescent_5NoOp:^GradientDescent_5/update_Variable_42/ApplyGradientDescent:^GradientDescent_5/update_Variable_43/ApplyGradientDescent:^GradientDescent_5/update_Variable_44/ApplyGradientDescent:^GradientDescent_5/update_Variable_45/ApplyGradientDescent
Q
Placeholder_37Placeholder*
_output_shapes
:*
dtype0*
shape: 
`
Placeholder_38Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€@
`
Placeholder_39Placeholder*
shape: *
dtype0*'
_output_shapes
:€€€€€€€€€

я
Merge_4/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4l1/outputs_3l2/outputs_3loss_5*
N*
_output_shapes
: 
»
init_1NoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign^Variable_44/Assign^Variable_45/Assign
Q
Placeholder_40Placeholder*
_output_shapes
:*
shape: *
dtype0
`
Placeholder_41Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€@
`
Placeholder_42Placeholder*'
_output_shapes
:€€€€€€€€€
*
dtype0*
shape: 
g
random_normal_23/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   2   
Z
random_normal_23/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_23/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
§
%random_normal_23/RandomStandardNormalRandomStandardNormalrandom_normal_23/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_23/mulMul%random_normal_23/RandomStandardNormalrandom_normal_23/stddev*
_output_shapes

:@2*
T0
m
random_normal_23Addrandom_normal_23/mulrandom_normal_23/mean*
_output_shapes

:@2*
T0

Variable_46
VariableV2*
shape
:@2*
shared_name *
dtype0*
_output_shapes

:@2*
	container 
≠
Variable_46/AssignAssignVariable_46random_normal_23*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@2*
_class
loc:@Variable_46
r
Variable_46/readIdentityVariable_46*
_class
loc:@Variable_46*
_output_shapes

:@2*
T0
]
zeros_23Const*
dtype0*
_output_shapes

:2*
valueB2*    
M
add_46/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_46Addzeros_23add_46/y*
T0*
_output_shapes

:2

Variable_47
VariableV2*
_output_shapes

:2*
	container *
shape
:2*
dtype0*
shared_name 
£
Variable_47/AssignAssignVariable_47add_46*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2*
_class
loc:@Variable_47
r
Variable_47/readIdentityVariable_47*
_class
loc:@Variable_47*
_output_shapes

:2*
T0
Н
	MatMul_23MatMulPlaceholder_41Variable_46/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
\
add_47Add	MatMul_23Variable_47/read*'
_output_shapes
:€€€€€€€€€2*
T0
Y
dropout_21/keep_probConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
I
Tanh_14Tanhadd_47*'
_output_shapes
:€€€€€€€€€2*
T0
]
l1/outputs_4/tagConst*
_output_shapes
: *
dtype0*
valueB Bl1/outputs_4
\
l1/outputs_4HistogramSummaryl1/outputs_4/tagTanh_14*
T0*
_output_shapes
: 
g
random_normal_24/shapeConst*
valueB"2   
   *
dtype0*
_output_shapes
:
Z
random_normal_24/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_24/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_24/RandomStandardNormalRandomStandardNormalrandom_normal_24/shape*
_output_shapes

:2
*
seed2 *
dtype0*
T0*

seed 
Д
random_normal_24/mulMul%random_normal_24/RandomStandardNormalrandom_normal_24/stddev*
T0*
_output_shapes

:2

m
random_normal_24Addrandom_normal_24/mulrandom_normal_24/mean*
T0*
_output_shapes

:2


Variable_48
VariableV2*
shape
:2
*
shared_name *
dtype0*
_output_shapes

:2
*
	container 
≠
Variable_48/AssignAssignVariable_48random_normal_24*
_output_shapes

:2
*
validate_shape(*
_class
loc:@Variable_48*
T0*
use_locking(
r
Variable_48/readIdentityVariable_48*
_output_shapes

:2
*
_class
loc:@Variable_48*
T0
]
zeros_24Const*
valueB
*    *
_output_shapes

:
*
dtype0
M
add_48/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
J
add_48Addzeros_24add_48/y*
_output_shapes

:
*
T0

Variable_49
VariableV2*
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*
	container 
£
Variable_49/AssignAssignVariable_49add_48*
_class
loc:@Variable_49*
_output_shapes

:
*
T0*
validate_shape(*
use_locking(
r
Variable_49/readIdentityVariable_49*
T0*
_output_shapes

:
*
_class
loc:@Variable_49
Ж
	MatMul_24MatMulTanh_14Variable_48/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_49Add	MatMul_24Variable_49/read*
T0*'
_output_shapes
:€€€€€€€€€

Y
dropout_22/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_7Softmaxadd_49*
T0*'
_output_shapes
:€€€€€€€€€

]
l2/outputs_4/tagConst*
valueB Bl2/outputs_4*
dtype0*
_output_shapes
: 
^
l2/outputs_4HistogramSummaryl2/outputs_4/tag	Softmax_7*
T0*
_output_shapes
: 
I
Log_7Log	Softmax_7*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_7MulPlaceholder_42Log_7*'
_output_shapes
:€€€€€€€€€
*
T0
a
Sum_7/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
w
Sum_7Summul_7Sum_7/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_7NegSum_7*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_7Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_7MeanNeg_7Const_7*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
loss_6/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_6
M
loss_6ScalarSummaryloss_6/tagsMean_7*
T0*
_output_shapes
: 
T
gradients_6/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
V
gradients_6/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
_
gradients_6/FillFillgradients_6/Shapegradients_6/Const*
T0*
_output_shapes
: 
o
%gradients_6/Mean_7_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
Ц
gradients_6/Mean_7_grad/ReshapeReshapegradients_6/Fill%gradients_6/Mean_7_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_6/Mean_7_grad/ShapeShapeNeg_7*
out_type0*
_output_shapes
:*
T0
§
gradients_6/Mean_7_grad/TileTilegradients_6/Mean_7_grad/Reshapegradients_6/Mean_7_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
d
gradients_6/Mean_7_grad/Shape_1ShapeNeg_7*
_output_shapes
:*
out_type0*
T0
b
gradients_6/Mean_7_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_6/Mean_7_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
Ґ
gradients_6/Mean_7_grad/ProdProdgradients_6/Mean_7_grad/Shape_1gradients_6/Mean_7_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_6/Mean_7_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
¶
gradients_6/Mean_7_grad/Prod_1Prodgradients_6/Mean_7_grad/Shape_2gradients_6/Mean_7_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_6/Mean_7_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
О
gradients_6/Mean_7_grad/MaximumMaximumgradients_6/Mean_7_grad/Prod_1!gradients_6/Mean_7_grad/Maximum/y*
T0*
_output_shapes
: 
М
 gradients_6/Mean_7_grad/floordivFloorDivgradients_6/Mean_7_grad/Prodgradients_6/Mean_7_grad/Maximum*
_output_shapes
: *
T0
v
gradients_6/Mean_7_grad/CastCast gradients_6/Mean_7_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
Ф
gradients_6/Mean_7_grad/truedivRealDivgradients_6/Mean_7_grad/Tilegradients_6/Mean_7_grad/Cast*
T0*#
_output_shapes
:€€€€€€€€€
p
gradients_6/Neg_7_grad/NegNeggradients_6/Mean_7_grad/truediv*
T0*#
_output_shapes
:€€€€€€€€€
a
gradients_6/Sum_7_grad/ShapeShapemul_7*
_output_shapes
:*
out_type0*
T0
]
gradients_6/Sum_7_grad/SizeConst*
value	B :*
_output_shapes
: *
dtype0
|
gradients_6/Sum_7_grad/addAddSum_7/reduction_indicesgradients_6/Sum_7_grad/Size*
_output_shapes
:*
T0
Д
gradients_6/Sum_7_grad/modFloorModgradients_6/Sum_7_grad/addgradients_6/Sum_7_grad/Size*
T0*
_output_shapes
:
h
gradients_6/Sum_7_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:
d
"gradients_6/Sum_7_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"gradients_6/Sum_7_grad/range/deltaConst*
value	B :*
_output_shapes
: *
dtype0
≤
gradients_6/Sum_7_grad/rangeRange"gradients_6/Sum_7_grad/range/startgradients_6/Sum_7_grad/Size"gradients_6/Sum_7_grad/range/delta*
_output_shapes
:*

Tidx0
c
!gradients_6/Sum_7_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
Л
gradients_6/Sum_7_grad/FillFillgradients_6/Sum_7_grad/Shape_1!gradients_6/Sum_7_grad/Fill/value*
_output_shapes
:*
T0
б
$gradients_6/Sum_7_grad/DynamicStitchDynamicStitchgradients_6/Sum_7_grad/rangegradients_6/Sum_7_grad/modgradients_6/Sum_7_grad/Shapegradients_6/Sum_7_grad/Fill*#
_output_shapes
:€€€€€€€€€*
T0*
N
b
 gradients_6/Sum_7_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_6/Sum_7_grad/MaximumMaximum$gradients_6/Sum_7_grad/DynamicStitch gradients_6/Sum_7_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
О
gradients_6/Sum_7_grad/floordivFloorDivgradients_6/Sum_7_grad/Shapegradients_6/Sum_7_grad/Maximum*
T0*
_output_shapes
:
Ь
gradients_6/Sum_7_grad/ReshapeReshapegradients_6/Neg_7_grad/Neg$gradients_6/Sum_7_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
®
gradients_6/Sum_7_grad/TileTilegradients_6/Sum_7_grad/Reshapegradients_6/Sum_7_grad/floordiv*'
_output_shapes
:€€€€€€€€€
*
T0*

Tmultiples0
j
gradients_6/mul_7_grad/ShapeShapePlaceholder_42*
out_type0*
_output_shapes
:*
T0
c
gradients_6/mul_7_grad/Shape_1ShapeLog_7*
out_type0*
_output_shapes
:*
T0
ј
,gradients_6/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/mul_7_grad/Shapegradients_6/mul_7_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_6/mul_7_grad/mulMulgradients_6/Sum_7_grad/TileLog_7*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_6/mul_7_grad/SumSumgradients_6/mul_7_grad/mul,gradients_6/mul_7_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_6/mul_7_grad/ReshapeReshapegradients_6/mul_7_grad/Sumgradients_6/mul_7_grad/Shape*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
В
gradients_6/mul_7_grad/mul_1MulPlaceholder_42gradients_6/Sum_7_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_6/mul_7_grad/Sum_1Sumgradients_6/mul_7_grad/mul_1.gradients_6/mul_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
©
 gradients_6/mul_7_grad/Reshape_1Reshapegradients_6/mul_7_grad/Sum_1gradients_6/mul_7_grad/Shape_1*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
s
'gradients_6/mul_7_grad/tuple/group_depsNoOp^gradients_6/mul_7_grad/Reshape!^gradients_6/mul_7_grad/Reshape_1
к
/gradients_6/mul_7_grad/tuple/control_dependencyIdentitygradients_6/mul_7_grad/Reshape(^gradients_6/mul_7_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_6/mul_7_grad/Reshape*
T0
р
1gradients_6/mul_7_grad/tuple/control_dependency_1Identity gradients_6/mul_7_grad/Reshape_1(^gradients_6/mul_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_6/mul_7_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_6/Log_7_grad/Reciprocal
Reciprocal	Softmax_72^gradients_6/mul_7_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:€€€€€€€€€

©
gradients_6/Log_7_grad/mulMul1gradients_6/mul_7_grad/tuple/control_dependency_1!gradients_6/Log_7_grad/Reciprocal*'
_output_shapes
:€€€€€€€€€
*
T0
~
gradients_6/Softmax_7_grad/mulMulgradients_6/Log_7_grad/mul	Softmax_7*'
_output_shapes
:€€€€€€€€€
*
T0
z
0gradients_6/Softmax_7_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
¬
gradients_6/Softmax_7_grad/SumSumgradients_6/Softmax_7_grad/mul0gradients_6/Softmax_7_grad/Sum/reduction_indices*#
_output_shapes
:€€€€€€€€€*
T0*
	keep_dims( *

Tidx0
y
(gradients_6/Softmax_7_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   
Ј
"gradients_6/Softmax_7_grad/ReshapeReshapegradients_6/Softmax_7_grad/Sum(gradients_6/Softmax_7_grad/Reshape/shape*
Tshape0*'
_output_shapes
:€€€€€€€€€*
T0
Ч
gradients_6/Softmax_7_grad/subSubgradients_6/Log_7_grad/mul"gradients_6/Softmax_7_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
Д
 gradients_6/Softmax_7_grad/mul_1Mulgradients_6/Softmax_7_grad/sub	Softmax_7*
T0*'
_output_shapes
:€€€€€€€€€

f
gradients_6/add_49_grad/ShapeShape	MatMul_24*
_output_shapes
:*
out_type0*
T0
p
gradients_6/add_49_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   
   
√
-gradients_6/add_49_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/add_49_grad/Shapegradients_6/add_49_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_6/add_49_grad/SumSum gradients_6/Softmax_7_grad/mul_1-gradients_6/add_49_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_6/add_49_grad/ReshapeReshapegradients_6/add_49_grad/Sumgradients_6/add_49_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
Ј
gradients_6/add_49_grad/Sum_1Sum gradients_6/Softmax_7_grad/mul_1/gradients_6/add_49_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
£
!gradients_6/add_49_grad/Reshape_1Reshapegradients_6/add_49_grad/Sum_1gradients_6/add_49_grad/Shape_1*
T0*
_output_shapes

:
*
Tshape0
v
(gradients_6/add_49_grad/tuple/group_depsNoOp ^gradients_6/add_49_grad/Reshape"^gradients_6/add_49_grad/Reshape_1
о
0gradients_6/add_49_grad/tuple/control_dependencyIdentitygradients_6/add_49_grad/Reshape)^gradients_6/add_49_grad/tuple/group_deps*2
_class(
&$loc:@gradients_6/add_49_grad/Reshape*'
_output_shapes
:€€€€€€€€€
*
T0
л
2gradients_6/add_49_grad/tuple/control_dependency_1Identity!gradients_6/add_49_grad/Reshape_1)^gradients_6/add_49_grad/tuple/group_deps*
_output_shapes

:
*4
_class*
(&loc:@gradients_6/add_49_grad/Reshape_1*
T0
«
!gradients_6/MatMul_24_grad/MatMulMatMul0gradients_6/add_49_grad/tuple/control_dependencyVariable_48/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
Ј
#gradients_6/MatMul_24_grad/MatMul_1MatMulTanh_140gradients_6/add_49_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:2
*
transpose_a(
}
+gradients_6/MatMul_24_grad/tuple/group_depsNoOp"^gradients_6/MatMul_24_grad/MatMul$^gradients_6/MatMul_24_grad/MatMul_1
ш
3gradients_6/MatMul_24_grad/tuple/control_dependencyIdentity!gradients_6/MatMul_24_grad/MatMul,^gradients_6/MatMul_24_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_6/MatMul_24_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
х
5gradients_6/MatMul_24_grad/tuple/control_dependency_1Identity#gradients_6/MatMul_24_grad/MatMul_1,^gradients_6/MatMul_24_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_6/MatMul_24_grad/MatMul_1*
_output_shapes

:2

Э
!gradients_6/Tanh_14_grad/TanhGradTanhGradTanh_143gradients_6/MatMul_24_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€2*
T0
f
gradients_6/add_47_grad/ShapeShape	MatMul_23*
T0*
out_type0*
_output_shapes
:
p
gradients_6/add_47_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"   2   
√
-gradients_6/add_47_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_6/add_47_grad/Shapegradients_6/add_47_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
gradients_6/add_47_grad/SumSum!gradients_6/Tanh_14_grad/TanhGrad-gradients_6/add_47_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
¶
gradients_6/add_47_grad/ReshapeReshapegradients_6/add_47_grad/Sumgradients_6/add_47_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€2
Є
gradients_6/add_47_grad/Sum_1Sum!gradients_6/Tanh_14_grad/TanhGrad/gradients_6/add_47_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_6/add_47_grad/Reshape_1Reshapegradients_6/add_47_grad/Sum_1gradients_6/add_47_grad/Shape_1*
_output_shapes

:2*
Tshape0*
T0
v
(gradients_6/add_47_grad/tuple/group_depsNoOp ^gradients_6/add_47_grad/Reshape"^gradients_6/add_47_grad/Reshape_1
о
0gradients_6/add_47_grad/tuple/control_dependencyIdentitygradients_6/add_47_grad/Reshape)^gradients_6/add_47_grad/tuple/group_deps*2
_class(
&$loc:@gradients_6/add_47_grad/Reshape*'
_output_shapes
:€€€€€€€€€2*
T0
л
2gradients_6/add_47_grad/tuple/control_dependency_1Identity!gradients_6/add_47_grad/Reshape_1)^gradients_6/add_47_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_6/add_47_grad/Reshape_1*
_output_shapes

:2
«
!gradients_6/MatMul_23_grad/MatMulMatMul0gradients_6/add_47_grad/tuple/control_dependencyVariable_46/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
T0
Њ
#gradients_6/MatMul_23_grad/MatMul_1MatMulPlaceholder_410gradients_6/add_47_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_6/MatMul_23_grad/tuple/group_depsNoOp"^gradients_6/MatMul_23_grad/MatMul$^gradients_6/MatMul_23_grad/MatMul_1
ш
3gradients_6/MatMul_23_grad/tuple/control_dependencyIdentity!gradients_6/MatMul_23_grad/MatMul,^gradients_6/MatMul_23_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*4
_class*
(&loc:@gradients_6/MatMul_23_grad/MatMul*
T0
х
5gradients_6/MatMul_23_grad/tuple/control_dependency_1Identity#gradients_6/MatMul_23_grad/MatMul_1,^gradients_6/MatMul_23_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_6/MatMul_23_grad/MatMul_1*
_output_shapes

:@2
d
GradientDescent_6/learning_rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Т
9GradientDescent_6/update_Variable_46/ApplyGradientDescentApplyGradientDescentVariable_46GradientDescent_6/learning_rate5gradients_6/MatMul_23_grad/tuple/control_dependency_1*
_class
loc:@Variable_46*
_output_shapes

:@2*
T0*
use_locking( 
П
9GradientDescent_6/update_Variable_47/ApplyGradientDescentApplyGradientDescentVariable_47GradientDescent_6/learning_rate2gradients_6/add_47_grad/tuple/control_dependency_1*
_class
loc:@Variable_47*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_6/update_Variable_48/ApplyGradientDescentApplyGradientDescentVariable_48GradientDescent_6/learning_rate5gradients_6/MatMul_24_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@Variable_48*
_output_shapes

:2

П
9GradientDescent_6/update_Variable_49/ApplyGradientDescentApplyGradientDescentVariable_49GradientDescent_6/learning_rate2gradients_6/add_49_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:
*
_class
loc:@Variable_49
Й
GradientDescent_6NoOp:^GradientDescent_6/update_Variable_46/ApplyGradientDescent:^GradientDescent_6/update_Variable_47/ApplyGradientDescent:^GradientDescent_6/update_Variable_48/ApplyGradientDescent:^GradientDescent_6/update_Variable_49/ApplyGradientDescent
Г
Merge_5/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4l1/outputs_3l2/outputs_3loss_5l1/outputs_4l2/outputs_4loss_6*
N*
_output_shapes
: 
Ь
init_2NoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^Variable_16/Assign^Variable_17/Assign^Variable_18/Assign^Variable_19/Assign^Variable_20/Assign^Variable_21/Assign^Variable_22/Assign^Variable_23/Assign^Variable_24/Assign^Variable_25/Assign^Variable_26/Assign^Variable_27/Assign^Variable_28/Assign^Variable_29/Assign^Variable_30/Assign^Variable_31/Assign^Variable_32/Assign^Variable_33/Assign^Variable_34/Assign^Variable_35/Assign^Variable_36/Assign^Variable_37/Assign^Variable_38/Assign^Variable_39/Assign^Variable_40/Assign^Variable_41/Assign^Variable_42/Assign^Variable_43/Assign^Variable_44/Assign^Variable_45/Assign^Variable_46/Assign^Variable_47/Assign^Variable_48/Assign^Variable_49/Assign
Q
Placeholder_43Placeholder*
_output_shapes
:*
dtype0*
shape: 
`
Placeholder_44Placeholder*'
_output_shapes
:€€€€€€€€€@*
shape: *
dtype0
`
Placeholder_45Placeholder*
dtype0*
shape: *'
_output_shapes
:€€€€€€€€€

g
random_normal_25/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   2   
Z
random_normal_25/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
random_normal_25/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
§
%random_normal_25/RandomStandardNormalRandomStandardNormalrandom_normal_25/shape*
dtype0*

seed *
T0*
_output_shapes

:@2*
seed2 
Д
random_normal_25/mulMul%random_normal_25/RandomStandardNormalrandom_normal_25/stddev*
T0*
_output_shapes

:@2
m
random_normal_25Addrandom_normal_25/mulrandom_normal_25/mean*
T0*
_output_shapes

:@2

Variable_50
VariableV2*
shared_name *
dtype0*
shape
:@2*
_output_shapes

:@2*
	container 
≠
Variable_50/AssignAssignVariable_50random_normal_25*
use_locking(*
T0*
_class
loc:@Variable_50*
validate_shape(*
_output_shapes

:@2
r
Variable_50/readIdentityVariable_50*
_output_shapes

:@2*
_class
loc:@Variable_50*
T0
]
zeros_25Const*
valueB2*    *
dtype0*
_output_shapes

:2
M
add_50/yConst*
valueB
 *Ќћћ=*
_output_shapes
: *
dtype0
J
add_50Addzeros_25add_50/y*
T0*
_output_shapes

:2

Variable_51
VariableV2*
shape
:2*
shared_name *
dtype0*
_output_shapes

:2*
	container 
£
Variable_51/AssignAssignVariable_51add_50*
use_locking(*
T0*
_class
loc:@Variable_51*
validate_shape(*
_output_shapes

:2
r
Variable_51/readIdentityVariable_51*
_class
loc:@Variable_51*
_output_shapes

:2*
T0
Н
	MatMul_25MatMulPlaceholder_44Variable_50/read*
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€2*
transpose_a( 
\
add_51Add	MatMul_25Variable_51/read*
T0*'
_output_shapes
:€€€€€€€€€2
Y
dropout_23/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
I
Tanh_15Tanhadd_51*'
_output_shapes
:€€€€€€€€€2*
T0
]
l1/outputs_5/tagConst*
dtype0*
_output_shapes
: *
valueB Bl1/outputs_5
\
l1/outputs_5HistogramSummaryl1/outputs_5/tagTanh_15*
T0*
_output_shapes
: 
g
random_normal_26/shapeConst*
valueB"2   
   *
_output_shapes
:*
dtype0
Z
random_normal_26/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
\
random_normal_26/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
§
%random_normal_26/RandomStandardNormalRandomStandardNormalrandom_normal_26/shape*
_output_shapes

:2
*
seed2 *
T0*

seed *
dtype0
Д
random_normal_26/mulMul%random_normal_26/RandomStandardNormalrandom_normal_26/stddev*
_output_shapes

:2
*
T0
m
random_normal_26Addrandom_normal_26/mulrandom_normal_26/mean*
T0*
_output_shapes

:2


Variable_52
VariableV2*
_output_shapes

:2
*
	container *
shape
:2
*
dtype0*
shared_name 
≠
Variable_52/AssignAssignVariable_52random_normal_26*
use_locking(*
validate_shape(*
T0*
_output_shapes

:2
*
_class
loc:@Variable_52
r
Variable_52/readIdentityVariable_52*
_class
loc:@Variable_52*
_output_shapes

:2
*
T0
]
zeros_26Const*
valueB
*    *
dtype0*
_output_shapes

:

M
add_52/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=
J
add_52Addzeros_26add_52/y*
_output_shapes

:
*
T0

Variable_53
VariableV2*
shared_name *
dtype0*
shape
:
*
_output_shapes

:
*
	container 
£
Variable_53/AssignAssignVariable_53add_52*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
*
_class
loc:@Variable_53
r
Variable_53/readIdentityVariable_53*
_output_shapes

:
*
_class
loc:@Variable_53*
T0
Ж
	MatMul_26MatMulTanh_15Variable_52/read*
transpose_b( *'
_output_shapes
:€€€€€€€€€
*
transpose_a( *
T0
\
add_53Add	MatMul_26Variable_53/read*'
_output_shapes
:€€€€€€€€€
*
T0
Y
dropout_24/keep_probConst*
valueB
 *  А?*
_output_shapes
: *
dtype0
N
	Softmax_8Softmaxadd_53*'
_output_shapes
:€€€€€€€€€
*
T0
]
l2/outputs_5/tagConst*
valueB Bl2/outputs_5*
_output_shapes
: *
dtype0
^
l2/outputs_5HistogramSummaryl2/outputs_5/tag	Softmax_8*
_output_shapes
: *
T0
I
Log_8Log	Softmax_8*
T0*'
_output_shapes
:€€€€€€€€€

U
mul_8MulPlaceholder_45Log_8*
T0*'
_output_shapes
:€€€€€€€€€

a
Sum_8/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
w
Sum_8Summul_8Sum_8/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
A
Neg_8NegSum_8*#
_output_shapes
:€€€€€€€€€*
T0
Q
Const_8Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_8MeanNeg_8Const_8*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
R
loss_7/tagsConst*
_output_shapes
: *
dtype0*
valueB Bloss_7
M
loss_7ScalarSummaryloss_7/tagsMean_8*
_output_shapes
: *
T0
T
gradients_7/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
V
gradients_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
_
gradients_7/FillFillgradients_7/Shapegradients_7/Const*
T0*
_output_shapes
: 
o
%gradients_7/Mean_8_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
Ц
gradients_7/Mean_8_grad/ReshapeReshapegradients_7/Fill%gradients_7/Mean_8_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_7/Mean_8_grad/ShapeShapeNeg_8*
out_type0*
_output_shapes
:*
T0
§
gradients_7/Mean_8_grad/TileTilegradients_7/Mean_8_grad/Reshapegradients_7/Mean_8_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:€€€€€€€€€
d
gradients_7/Mean_8_grad/Shape_1ShapeNeg_8*
T0*
_output_shapes
:*
out_type0
b
gradients_7/Mean_8_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
g
gradients_7/Mean_8_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
Ґ
gradients_7/Mean_8_grad/ProdProdgradients_7/Mean_8_grad/Shape_1gradients_7/Mean_8_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
i
gradients_7/Mean_8_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
¶
gradients_7/Mean_8_grad/Prod_1Prodgradients_7/Mean_8_grad/Shape_2gradients_7/Mean_8_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_7/Mean_8_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0
О
gradients_7/Mean_8_grad/MaximumMaximumgradients_7/Mean_8_grad/Prod_1!gradients_7/Mean_8_grad/Maximum/y*
_output_shapes
: *
T0
М
 gradients_7/Mean_8_grad/floordivFloorDivgradients_7/Mean_8_grad/Prodgradients_7/Mean_8_grad/Maximum*
_output_shapes
: *
T0
v
gradients_7/Mean_8_grad/CastCast gradients_7/Mean_8_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
Ф
gradients_7/Mean_8_grad/truedivRealDivgradients_7/Mean_8_grad/Tilegradients_7/Mean_8_grad/Cast*#
_output_shapes
:€€€€€€€€€*
T0
p
gradients_7/Neg_8_grad/NegNeggradients_7/Mean_8_grad/truediv*#
_output_shapes
:€€€€€€€€€*
T0
a
gradients_7/Sum_8_grad/ShapeShapemul_8*
_output_shapes
:*
out_type0*
T0
]
gradients_7/Sum_8_grad/SizeConst*
_output_shapes
: *
dtype0*
value	B :
|
gradients_7/Sum_8_grad/addAddSum_8/reduction_indicesgradients_7/Sum_8_grad/Size*
T0*
_output_shapes
:
Д
gradients_7/Sum_8_grad/modFloorModgradients_7/Sum_8_grad/addgradients_7/Sum_8_grad/Size*
T0*
_output_shapes
:
h
gradients_7/Sum_8_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
d
"gradients_7/Sum_8_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"gradients_7/Sum_8_grad/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
≤
gradients_7/Sum_8_grad/rangeRange"gradients_7/Sum_8_grad/range/startgradients_7/Sum_8_grad/Size"gradients_7/Sum_8_grad/range/delta*

Tidx0*
_output_shapes
:
c
!gradients_7/Sum_8_grad/Fill/valueConst*
value	B :*
_output_shapes
: *
dtype0
Л
gradients_7/Sum_8_grad/FillFillgradients_7/Sum_8_grad/Shape_1!gradients_7/Sum_8_grad/Fill/value*
T0*
_output_shapes
:
б
$gradients_7/Sum_8_grad/DynamicStitchDynamicStitchgradients_7/Sum_8_grad/rangegradients_7/Sum_8_grad/modgradients_7/Sum_8_grad/Shapegradients_7/Sum_8_grad/Fill*#
_output_shapes
:€€€€€€€€€*
N*
T0
b
 gradients_7/Sum_8_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
Я
gradients_7/Sum_8_grad/MaximumMaximum$gradients_7/Sum_8_grad/DynamicStitch gradients_7/Sum_8_grad/Maximum/y*#
_output_shapes
:€€€€€€€€€*
T0
О
gradients_7/Sum_8_grad/floordivFloorDivgradients_7/Sum_8_grad/Shapegradients_7/Sum_8_grad/Maximum*
T0*
_output_shapes
:
Ь
gradients_7/Sum_8_grad/ReshapeReshapegradients_7/Neg_8_grad/Neg$gradients_7/Sum_8_grad/DynamicStitch*
T0*
_output_shapes
:*
Tshape0
®
gradients_7/Sum_8_grad/TileTilegradients_7/Sum_8_grad/Reshapegradients_7/Sum_8_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€

j
gradients_7/mul_8_grad/ShapeShapePlaceholder_45*
T0*
_output_shapes
:*
out_type0
c
gradients_7/mul_8_grad/Shape_1ShapeLog_8*
out_type0*
_output_shapes
:*
T0
ј
,gradients_7/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/mul_8_grad/Shapegradients_7/mul_8_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
w
gradients_7/mul_8_grad/mulMulgradients_7/Sum_8_grad/TileLog_8*'
_output_shapes
:€€€€€€€€€
*
T0
Ђ
gradients_7/mul_8_grad/SumSumgradients_7/mul_8_grad/mul,gradients_7/mul_8_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
gradients_7/mul_8_grad/ReshapeReshapegradients_7/mul_8_grad/Sumgradients_7/mul_8_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€
*
T0
В
gradients_7/mul_8_grad/mul_1MulPlaceholder_45gradients_7/Sum_8_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€

±
gradients_7/mul_8_grad/Sum_1Sumgradients_7/mul_8_grad/mul_1.gradients_7/mul_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
©
 gradients_7/mul_8_grad/Reshape_1Reshapegradients_7/mul_8_grad/Sum_1gradients_7/mul_8_grad/Shape_1*'
_output_shapes
:€€€€€€€€€
*
Tshape0*
T0
s
'gradients_7/mul_8_grad/tuple/group_depsNoOp^gradients_7/mul_8_grad/Reshape!^gradients_7/mul_8_grad/Reshape_1
к
/gradients_7/mul_8_grad/tuple/control_dependencyIdentitygradients_7/mul_8_grad/Reshape(^gradients_7/mul_8_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€
*1
_class'
%#loc:@gradients_7/mul_8_grad/Reshape*
T0
р
1gradients_7/mul_8_grad/tuple/control_dependency_1Identity gradients_7/mul_8_grad/Reshape_1(^gradients_7/mul_8_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_7/mul_8_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€

†
!gradients_7/Log_8_grad/Reciprocal
Reciprocal	Softmax_82^gradients_7/mul_8_grad/tuple/control_dependency_1*'
_output_shapes
:€€€€€€€€€
*
T0
©
gradients_7/Log_8_grad/mulMul1gradients_7/mul_8_grad/tuple/control_dependency_1!gradients_7/Log_8_grad/Reciprocal*
T0*'
_output_shapes
:€€€€€€€€€

~
gradients_7/Softmax_8_grad/mulMulgradients_7/Log_8_grad/mul	Softmax_8*
T0*'
_output_shapes
:€€€€€€€€€

z
0gradients_7/Softmax_8_grad/Sum/reduction_indicesConst*
valueB:*
_output_shapes
:*
dtype0
¬
gradients_7/Softmax_8_grad/SumSumgradients_7/Softmax_8_grad/mul0gradients_7/Softmax_8_grad/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:€€€€€€€€€
y
(gradients_7/Softmax_8_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"€€€€   
Ј
"gradients_7/Softmax_8_grad/ReshapeReshapegradients_7/Softmax_8_grad/Sum(gradients_7/Softmax_8_grad/Reshape/shape*
T0*'
_output_shapes
:€€€€€€€€€*
Tshape0
Ч
gradients_7/Softmax_8_grad/subSubgradients_7/Log_8_grad/mul"gradients_7/Softmax_8_grad/Reshape*
T0*'
_output_shapes
:€€€€€€€€€

Д
 gradients_7/Softmax_8_grad/mul_1Mulgradients_7/Softmax_8_grad/sub	Softmax_8*'
_output_shapes
:€€€€€€€€€
*
T0
f
gradients_7/add_53_grad/ShapeShape	MatMul_26*
T0*
_output_shapes
:*
out_type0
p
gradients_7/add_53_grad/Shape_1Const*
valueB"   
   *
_output_shapes
:*
dtype0
√
-gradients_7/add_53_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/add_53_grad/Shapegradients_7/add_53_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
≥
gradients_7/add_53_grad/SumSum gradients_7/Softmax_8_grad/mul_1-gradients_7/add_53_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_7/add_53_grad/ReshapeReshapegradients_7/add_53_grad/Sumgradients_7/add_53_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€
*
Tshape0
Ј
gradients_7/add_53_grad/Sum_1Sum gradients_7/Softmax_8_grad/mul_1/gradients_7/add_53_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_7/add_53_grad/Reshape_1Reshapegradients_7/add_53_grad/Sum_1gradients_7/add_53_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:

v
(gradients_7/add_53_grad/tuple/group_depsNoOp ^gradients_7/add_53_grad/Reshape"^gradients_7/add_53_grad/Reshape_1
о
0gradients_7/add_53_grad/tuple/control_dependencyIdentitygradients_7/add_53_grad/Reshape)^gradients_7/add_53_grad/tuple/group_deps*
T0*'
_output_shapes
:€€€€€€€€€
*2
_class(
&$loc:@gradients_7/add_53_grad/Reshape
л
2gradients_7/add_53_grad/tuple/control_dependency_1Identity!gradients_7/add_53_grad/Reshape_1)^gradients_7/add_53_grad/tuple/group_deps*4
_class*
(&loc:@gradients_7/add_53_grad/Reshape_1*
_output_shapes

:
*
T0
«
!gradients_7/MatMul_26_grad/MatMulMatMul0gradients_7/add_53_grad/tuple/control_dependencyVariable_52/read*
transpose_b(*'
_output_shapes
:€€€€€€€€€2*
transpose_a( *
T0
Ј
#gradients_7/MatMul_26_grad/MatMul_1MatMulTanh_150gradients_7/add_53_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:2
*
transpose_a(*
T0
}
+gradients_7/MatMul_26_grad/tuple/group_depsNoOp"^gradients_7/MatMul_26_grad/MatMul$^gradients_7/MatMul_26_grad/MatMul_1
ш
3gradients_7/MatMul_26_grad/tuple/control_dependencyIdentity!gradients_7/MatMul_26_grad/MatMul,^gradients_7/MatMul_26_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_7/MatMul_26_grad/MatMul*'
_output_shapes
:€€€€€€€€€2
х
5gradients_7/MatMul_26_grad/tuple/control_dependency_1Identity#gradients_7/MatMul_26_grad/MatMul_1,^gradients_7/MatMul_26_grad/tuple/group_deps*
T0*6
_class,
*(loc:@gradients_7/MatMul_26_grad/MatMul_1*
_output_shapes

:2

Э
!gradients_7/Tanh_15_grad/TanhGradTanhGradTanh_153gradients_7/MatMul_26_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€2
f
gradients_7/add_51_grad/ShapeShape	MatMul_25*
T0*
out_type0*
_output_shapes
:
p
gradients_7/add_51_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"   2   
√
-gradients_7/add_51_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_7/add_51_grad/Shapegradients_7/add_51_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
і
gradients_7/add_51_grad/SumSum!gradients_7/Tanh_15_grad/TanhGrad-gradients_7/add_51_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
¶
gradients_7/add_51_grad/ReshapeReshapegradients_7/add_51_grad/Sumgradients_7/add_51_grad/Shape*
Tshape0*'
_output_shapes
:€€€€€€€€€2*
T0
Є
gradients_7/add_51_grad/Sum_1Sum!gradients_7/Tanh_15_grad/TanhGrad/gradients_7/add_51_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
£
!gradients_7/add_51_grad/Reshape_1Reshapegradients_7/add_51_grad/Sum_1gradients_7/add_51_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:2
v
(gradients_7/add_51_grad/tuple/group_depsNoOp ^gradients_7/add_51_grad/Reshape"^gradients_7/add_51_grad/Reshape_1
о
0gradients_7/add_51_grad/tuple/control_dependencyIdentitygradients_7/add_51_grad/Reshape)^gradients_7/add_51_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€2*2
_class(
&$loc:@gradients_7/add_51_grad/Reshape*
T0
л
2gradients_7/add_51_grad/tuple/control_dependency_1Identity!gradients_7/add_51_grad/Reshape_1)^gradients_7/add_51_grad/tuple/group_deps*
_output_shapes

:2*4
_class*
(&loc:@gradients_7/add_51_grad/Reshape_1*
T0
«
!gradients_7/MatMul_25_grad/MatMulMatMul0gradients_7/add_51_grad/tuple/control_dependencyVariable_50/read*
transpose_b(*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( 
Њ
#gradients_7/MatMul_25_grad/MatMul_1MatMulPlaceholder_440gradients_7/add_51_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@2*
transpose_a(
}
+gradients_7/MatMul_25_grad/tuple/group_depsNoOp"^gradients_7/MatMul_25_grad/MatMul$^gradients_7/MatMul_25_grad/MatMul_1
ш
3gradients_7/MatMul_25_grad/tuple/control_dependencyIdentity!gradients_7/MatMul_25_grad/MatMul,^gradients_7/MatMul_25_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*4
_class*
(&loc:@gradients_7/MatMul_25_grad/MatMul*
T0
х
5gradients_7/MatMul_25_grad/tuple/control_dependency_1Identity#gradients_7/MatMul_25_grad/MatMul_1,^gradients_7/MatMul_25_grad/tuple/group_deps*
_output_shapes

:@2*6
_class,
*(loc:@gradients_7/MatMul_25_grad/MatMul_1*
T0
d
GradientDescent_7/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
Т
9GradientDescent_7/update_Variable_50/ApplyGradientDescentApplyGradientDescentVariable_50GradientDescent_7/learning_rate5gradients_7/MatMul_25_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:@2*
_class
loc:@Variable_50
П
9GradientDescent_7/update_Variable_51/ApplyGradientDescentApplyGradientDescentVariable_51GradientDescent_7/learning_rate2gradients_7/add_51_grad/tuple/control_dependency_1*
_class
loc:@Variable_51*
_output_shapes

:2*
T0*
use_locking( 
Т
9GradientDescent_7/update_Variable_52/ApplyGradientDescentApplyGradientDescentVariable_52GradientDescent_7/learning_rate5gradients_7/MatMul_26_grad/tuple/control_dependency_1*
_class
loc:@Variable_52*
_output_shapes

:2
*
T0*
use_locking( 
П
9GradientDescent_7/update_Variable_53/ApplyGradientDescentApplyGradientDescentVariable_53GradientDescent_7/learning_rate2gradients_7/add_53_grad/tuple/control_dependency_1*
use_locking( *
T0*
_output_shapes

:
*
_class
loc:@Variable_53
Й
GradientDescent_7NoOp:^GradientDescent_7/update_Variable_50/ApplyGradientDescent:^GradientDescent_7/update_Variable_51/ApplyGradientDescent:^GradientDescent_7/update_Variable_52/ApplyGradientDescent:^GradientDescent_7/update_Variable_53/ApplyGradientDescent
І
Merge_6/MergeSummaryMergeSummarylossloss_1
l1/outputs
l2/outputsloss_2l1/outputs_1l2/outputs_1loss_3l1/outputs_2l2/outputs_2loss_4l1/outputs_3l2/outputs_3loss_5l1/outputs_4l2/outputs_4loss_6l1/outputs_5l2/outputs_5loss_7*
_output_shapes
: *
N""¶
train_opЩ
Ц
GradientDescent
GradientDescent_1
GradientDescent_2
GradientDescent_3
GradientDescent_4
GradientDescent_5
GradientDescent_6
GradientDescent_7"э
trainable_variablesев
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0
4
Variable_8:0Variable_8/AssignVariable_8/read:0
4
Variable_9:0Variable_9/AssignVariable_9/read:0
7
Variable_10:0Variable_10/AssignVariable_10/read:0
7
Variable_11:0Variable_11/AssignVariable_11/read:0
7
Variable_12:0Variable_12/AssignVariable_12/read:0
7
Variable_13:0Variable_13/AssignVariable_13/read:0
7
Variable_14:0Variable_14/AssignVariable_14/read:0
7
Variable_15:0Variable_15/AssignVariable_15/read:0
7
Variable_16:0Variable_16/AssignVariable_16/read:0
7
Variable_17:0Variable_17/AssignVariable_17/read:0
7
Variable_18:0Variable_18/AssignVariable_18/read:0
7
Variable_19:0Variable_19/AssignVariable_19/read:0
7
Variable_20:0Variable_20/AssignVariable_20/read:0
7
Variable_21:0Variable_21/AssignVariable_21/read:0
7
Variable_22:0Variable_22/AssignVariable_22/read:0
7
Variable_23:0Variable_23/AssignVariable_23/read:0
7
Variable_24:0Variable_24/AssignVariable_24/read:0
7
Variable_25:0Variable_25/AssignVariable_25/read:0
7
Variable_26:0Variable_26/AssignVariable_26/read:0
7
Variable_27:0Variable_27/AssignVariable_27/read:0
7
Variable_28:0Variable_28/AssignVariable_28/read:0
7
Variable_29:0Variable_29/AssignVariable_29/read:0
7
Variable_30:0Variable_30/AssignVariable_30/read:0
7
Variable_31:0Variable_31/AssignVariable_31/read:0
7
Variable_32:0Variable_32/AssignVariable_32/read:0
7
Variable_33:0Variable_33/AssignVariable_33/read:0
7
Variable_34:0Variable_34/AssignVariable_34/read:0
7
Variable_35:0Variable_35/AssignVariable_35/read:0
7
Variable_36:0Variable_36/AssignVariable_36/read:0
7
Variable_37:0Variable_37/AssignVariable_37/read:0
7
Variable_38:0Variable_38/AssignVariable_38/read:0
7
Variable_39:0Variable_39/AssignVariable_39/read:0
7
Variable_40:0Variable_40/AssignVariable_40/read:0
7
Variable_41:0Variable_41/AssignVariable_41/read:0
7
Variable_42:0Variable_42/AssignVariable_42/read:0
7
Variable_43:0Variable_43/AssignVariable_43/read:0
7
Variable_44:0Variable_44/AssignVariable_44/read:0
7
Variable_45:0Variable_45/AssignVariable_45/read:0
7
Variable_46:0Variable_46/AssignVariable_46/read:0
7
Variable_47:0Variable_47/AssignVariable_47/read:0
7
Variable_48:0Variable_48/AssignVariable_48/read:0
7
Variable_49:0Variable_49/AssignVariable_49/read:0
7
Variable_50:0Variable_50/AssignVariable_50/read:0
7
Variable_51:0Variable_51/AssignVariable_51/read:0
7
Variable_52:0Variable_52/AssignVariable_52/read:0
7
Variable_53:0Variable_53/AssignVariable_53/read:0"Ы
	summariesН
К
loss:0
loss_1:0
l1/outputs:0
l2/outputs:0
loss_2:0
l1/outputs_1:0
l2/outputs_1:0
loss_3:0
l1/outputs_2:0
l2/outputs_2:0
loss_4:0
l1/outputs_3:0
l2/outputs_3:0
loss_5:0
l1/outputs_4:0
l2/outputs_4:0
loss_6:0
l1/outputs_5:0
l2/outputs_5:0
loss_7:0"у
	variablesев
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0
4
Variable_8:0Variable_8/AssignVariable_8/read:0
4
Variable_9:0Variable_9/AssignVariable_9/read:0
7
Variable_10:0Variable_10/AssignVariable_10/read:0
7
Variable_11:0Variable_11/AssignVariable_11/read:0
7
Variable_12:0Variable_12/AssignVariable_12/read:0
7
Variable_13:0Variable_13/AssignVariable_13/read:0
7
Variable_14:0Variable_14/AssignVariable_14/read:0
7
Variable_15:0Variable_15/AssignVariable_15/read:0
7
Variable_16:0Variable_16/AssignVariable_16/read:0
7
Variable_17:0Variable_17/AssignVariable_17/read:0
7
Variable_18:0Variable_18/AssignVariable_18/read:0
7
Variable_19:0Variable_19/AssignVariable_19/read:0
7
Variable_20:0Variable_20/AssignVariable_20/read:0
7
Variable_21:0Variable_21/AssignVariable_21/read:0
7
Variable_22:0Variable_22/AssignVariable_22/read:0
7
Variable_23:0Variable_23/AssignVariable_23/read:0
7
Variable_24:0Variable_24/AssignVariable_24/read:0
7
Variable_25:0Variable_25/AssignVariable_25/read:0
7
Variable_26:0Variable_26/AssignVariable_26/read:0
7
Variable_27:0Variable_27/AssignVariable_27/read:0
7
Variable_28:0Variable_28/AssignVariable_28/read:0
7
Variable_29:0Variable_29/AssignVariable_29/read:0
7
Variable_30:0Variable_30/AssignVariable_30/read:0
7
Variable_31:0Variable_31/AssignVariable_31/read:0
7
Variable_32:0Variable_32/AssignVariable_32/read:0
7
Variable_33:0Variable_33/AssignVariable_33/read:0
7
Variable_34:0Variable_34/AssignVariable_34/read:0
7
Variable_35:0Variable_35/AssignVariable_35/read:0
7
Variable_36:0Variable_36/AssignVariable_36/read:0
7
Variable_37:0Variable_37/AssignVariable_37/read:0
7
Variable_38:0Variable_38/AssignVariable_38/read:0
7
Variable_39:0Variable_39/AssignVariable_39/read:0
7
Variable_40:0Variable_40/AssignVariable_40/read:0
7
Variable_41:0Variable_41/AssignVariable_41/read:0
7
Variable_42:0Variable_42/AssignVariable_42/read:0
7
Variable_43:0Variable_43/AssignVariable_43/read:0
7
Variable_44:0Variable_44/AssignVariable_44/read:0
7
Variable_45:0Variable_45/AssignVariable_45/read:0
7
Variable_46:0Variable_46/AssignVariable_46/read:0
7
Variable_47:0Variable_47/AssignVariable_47/read:0
7
Variable_48:0Variable_48/AssignVariable_48/read:0
7
Variable_49:0Variable_49/AssignVariable_49/read:0
7
Variable_50:0Variable_50/AssignVariable_50/read:0
7
Variable_51:0Variable_51/AssignVariable_51/read:0
7
Variable_52:0Variable_52/AssignVariable_52/read:0
7
Variable_53:0Variable_53/AssignVariable_53/read:0луx