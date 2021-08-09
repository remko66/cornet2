from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model



## Four simple functions to make a 2,5,5,2 resnet
#see base_ori for the original code as found all over stackexchange and other place.
#this if modified to be able to add bloakcs
#we can then train the whole network with added blocks (a progressive resnet) or only the last 2 blocks (a corrective neural net(cornet).
# the term cornet and corrective neural net are mine, see readme. But call it whatevr you like


def relu_bn(inputs):
    """
    Part of teh resnet block. Nothing changed here
    :param inputs:
    :return:
    """
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn



def residual_block(x, downsample, filters, kernel_size= 3):
    """
    the standard resnetblock. Nothing changed here!
    :param x:
    :param downsample:
    :param filters:
    :param kernel_size:
    :return:
    """
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same"
                  )(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out

def compile_model(model):
    """
    standard way to compile a keras network...nothing changed here as well!
    :param model:
    :return:
    """
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


def makeblocklist():
    """
    Make a list of blocks with true/false for downsample and nr of filters. Not all blocks are equal and to keep this experiment meaningfull will stick to the design
    :return:
    """
    blocklist=[]
    num_filters=64
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
           blocklist.append( ((j == 0 and i != 0),num_filters))
        num_filters *= 2
    return blocklist


def addBlock(x,curblock):
    """
    Add a resnet block to the network
    :param x: last layer
    :param curblock: blocknumber
    :return: new network
    """
    bl=makeblocklist()
    x=residual_block(x,downsample=bl[curblock][0],filters=bl[curblock][1])
    return x

def closeblock(t):
    """
    The way to end the network.
    :param t:
    :return: model
    """
    t = AveragePooling2D()(t)
    t = Flatten()(t)
    t = Dense(10, activation='softmax')(t)
    return t

def create_res_net():
    """
    Make a resnet with proper closure but only one residual block. rest added during training
    :return:  model
    """
    inputs = Input(shape=(32, 32, 3))
    num_filters = 64

    t = BatchNormalization()(inputs)
    t = Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    t=addBlock(t,0)
    outputs=closeblock(t)
    model = Model(inputs, outputs)
    compile_model(model)
    return model

def expand_res_net(model,curblock,trainold=True):
    """
    Add one extra block to the model
    :param model: current model
    :param curblock: current block number
    :param trainold: if yes then all old layers remain trainable, making this a progressive network. If no, only last 2 blocks can be trained. Making it a cornet)
    :return: model
    """
    inp=model.inputs
    for layer in model.layers[:-11]:
        layer.trainable=trainold
    x=model.layers[-4].output
    x=addBlock(x,curblock)
    x=closeblock(x)
    model = Model(inp, x)
    compile_model(model)
    model.summary()
    return model
