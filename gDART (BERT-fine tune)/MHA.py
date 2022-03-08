
# import keras
# from keras import backend as K
# from keras.layers import Layer
# import tensorflow as tf
# from keras import activations, initializers, constraints
# from keras.regularizers import l1, l2, l1_l2

#==========================
import tensorflow as tf

#==========================



class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, units = 100, Num_heads = 8, Num_combs = 6, output_units = 800, kernel_initializer=tf.keras.initializers.GlorotUniform(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4), kernel_constraint=None, use_bias=True, bias_initializer=tf.keras.initializers.GlorotUniform(), bias_regularizer=None, bias_constraint=None, **kwargs):
		self.units = units
		self.Num_heads = Num_heads
		self.Num_combs = Num_combs
		self.output_units = output_units
		self.supports_masking = True
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.use_bias = use_bias
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)
		self.kernel, self.b = None, None
		super(MultiHeadAttention, self).__init__(**kwargs)

	def build(self, input_shape):
		self.WQL = [[self.add_weight(name='Q_kernel_'+str(i)+str(j), shape=(input_shape[-1],self.units), initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable=True, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint) for j in range(self.Num_heads)] for i in range(self.Num_combs)]
		self.WKL = [[self.add_weight(name='K_kernel_'+str(i)+str(j), shape=(input_shape[-1],self.units), initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable=True, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint) for j in range(self.Num_heads)] for i in range(self.Num_combs)]
		self.WVL = [[self.add_weight(name='V_kernel_'+str(i)+str(j), shape=(input_shape[-1],self.units), initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable=True, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint) for j in range(self.Num_heads)] for i in range(self.Num_combs)]
		self.W_dotL = [[self.add_weight(name='general_dotProduct_'+str(i)+str(j), shape=(self.units,self.units), initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable=True, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint) for j in range(self.Num_heads)] for i in range(self.Num_combs)]
		self.WO = [self.add_weight(name="WO_weight"+str(i),shape=(self.output_units, self.units),initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.), trainable=True, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint) for i in range(self.Num_combs)]
		
		if self.use_bias:
			self.b = self.add_weight(shape=(input_shape[1], self.units), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, name='{}_b'.format(self.name),)
		
		super(MultiHeadAttention, self).build(input_shape)

	def call(self,inputs):
	
		#print ("kernel shape ", [ [y.shape for y in x] for x in self.WQL])	
	
		def MHA_f(i,j):
			query = tf.matmul(inputs, self.WQL[i][j])
			key = tf.matmul(inputs, self.WKL[i][j])
			value = tf.matmul(inputs, self.WVL[i][j])
			#scaled_attention_logits = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, tf.float32))
			query_W= tf.matmul(query,self.W_dotL[i][j])
			scaled_attention_logits = tf.matmul(query_W, key, transpose_b=True)
			attention_weights=tf.keras.activations.softmax(scaled_attention_logits, axis=-1)
			output = tf.matmul(attention_weights, value)	
			return output

		def comb(i):
			ZL = [MHA_f(i,j) for j in range(self.Num_heads)]
			Z = tf.concat(ZL, 2)
			Z = tf.matmul(Z,self.WO[i])
			return Z
		
		Z_final = []
		for i in range(self.Num_combs):
			Z_final.append(comb(i))
			
		if self.use_bias:
			Z_final=[x+self.b for x in Z_final]
		Z_final=[tf.nn.leaky_relu(x, alpha=0.1) for x in Z_final]##################################	
		#Z_final=[keras.activations.sigmoid(x) for x in Z_final]		
		return Z_final



	# def get_config(self):
	# 	config = super(MultiHeadAttention,self).get_config()
	# 	config.update({"units":self.units,"Num_heads":self.Num_heads,"Num_combs":self.Num_combs,"output_units":self.output_units})
	# 	return config
		
	# def compute_output_shape(self,input_shape):
	# 	l=[(input_shape[0],input_shape[1],self.units) for i in range(self.Num_combs)]
	# 	return l
		
	def compute_mask(self, inputs, mask=None):
		mask = super(MultiHeadAttention, self).compute_mask(inputs, mask)
		#if mask is None:
		return [None for i in range(self.Num_combs)]
		#return [mask for i in range(self.Num_combs)]
