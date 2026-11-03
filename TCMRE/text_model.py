# coding=utf-8
'''
This script is based on https://github.com/cjymz886/text_bert_cnn and https://github.com/google-research/bert.
'''
import  tensorflow as tf
import collections
from encodetoken import tokenization

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. 
          text_b: string. The untokenized text of the second sequence.
          label: string. The label of the example. This should be specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(examples,label_list, max_seq_length,tokenizer):
    """Loads a data file into a list of `InputBatch`s.
    Convert examples into token form as input of BERT model.
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    input_data=[]
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        if ex_index % 10000 == 0:
            tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]
        if ex_index < 3:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("guid: %s" % (example.guid))
            tf.compat.v1.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
            tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features = collections.OrderedDict()
        features["input_ids"] = input_ids
        features["input_mask"] = input_mask
        features["segment_ids"] = segment_ids
        features["label_ids"] =label_id
        input_data.append(features)

    return input_data

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def batch_iter(input_data,batch_size):
    """Batch feed the four tokens form variables of the sample into the model.s
    """
    batch_ids,batch_mask,batch_segment,batch_label=[],[],[],[]
    for features in input_data:
        if len(batch_ids) == batch_size:
            yield batch_ids,batch_mask,batch_segment,batch_label
            batch_ids, batch_mask, batch_segment, batch_label = [], [], [], []

        batch_ids.append(features['input_ids'])
        batch_mask.append(features['input_mask'])
        batch_segment.append(features['segment_ids'])
        batch_label.append(features['label_ids'])

    if len(batch_ids) != 0:
        yield batch_ids, batch_mask, batch_segment, batch_label


    
class TextMLP(object):

    def __init__(self,config):
        '''Get the hyperparameters and the five variables needed by the model, i.e. input_ids，input_mask，segment_ids，labels，keep_prob'''
        self.config=config
        tf.compat.v1.disable_eager_execution()
        self.input_ids=tf.compat.v1.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_ids')
        self.input_mask=tf.compat.v1.placeholder(tf.int64,shape=[None,self.config.seq_length],name='input_mask')
        self.segment_ids=tf.compat.v1.placeholder(tf.int64,shape=[None,self.config.seq_length],name='segment_ids')
        self.labels=tf.compat.v1.placeholder(tf.int64,shape=[None,],name='labels')
        self.keep_prob=tf.compat.v1.placeholder(tf.float32,name='dropout')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.MLP()

    def MLP(self):
        with tf.device('/cpu:0'):
            embedding = tf.compat.v1.get_variable('embedding', [50000, self.config.hidden_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_ids)
        with tf.name_scope('mlp'):
    # Flatten the BERT embeddings for MLP input
            embedding_inputs_flattened = tf.reshape(embedding_inputs, [-1, self.config.seq_length*self.config.hidden_dim])

    # Define dense layers
            dense_layers = []
            for idx, num_units in enumerate(self.config.filter_sizess):  # Assuming mlp_units is a list defining the number of units in each dense layer
                with tf.compat.v1.variable_scope(f"mlp_layer-{idx}", reuse=False):
                    dense = tf.compat.v1.layers.dense(embedding_inputs_flattened, num_units, activation=tf.nn.relu if idx < len(self.config.filter_sizess)-1 else None, name=f'dense_{idx}')
                    if idx < len(self.config.filter_sizess)-1:  # Apply dropout except for the last layer
                       dense = tf.compat.v1.nn.dropout(dense, self.keep_prob)
                dense_layers.append(dense)
                embedding_inputs_flattened = dense  # The output of one layer becomes the input to the next

    # The final output of the MLP
            outputs = dense_layers[-1]
            # outputs = []



        '''Add full connection layer and dropout layer'''
        with tf.name_scope('fc'):
            fc=tf.compat.v1.layers.dense(outputs,self.config.hidden_dim,name='fc1')
            fc = tf.compat.v1.nn.dropout(fc, rate=1-self.keep_prob)
            fc=tf.compat.v1.nn.relu(fc)

        '''logits'''
        with tf.name_scope('logits'):
            self.logits = tf.compat.v1.layers.dense(fc, self.config.num_labels, name='logits')
            self.prob = tf.compat.v1.nn.softmax(self.logits)
            self.y_pred_cls = tf.compat.v1.argmax(tf.compat.v1.nn.softmax(self.logits), 1)

        '''Calculate loss. Convert predicted labels into one hot form. '''
        with tf.name_scope('loss'):
            log_probs = tf.compat.v1.nn.log_softmax(self.logits, axis=-1)
            one_hot_labels = tf.compat.v1.one_hot(self.labels, depth=self.config.num_labels, dtype=tf.compat.v1.float32)
            per_example_loss = -tf.compat.v1.reduce_sum(one_hot_labels * log_probs, axis=-1)
            self.loss = tf.compat.v1.reduce_mean(per_example_loss)

        '''optimizer'''
        with tf.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.compat.v1.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        '''accuracy'''
        with tf.name_scope('accuracy'):
            correct_pred = tf.compat.v1.equal(self.labels, self.y_pred_cls)
            self.acc=tf.compat.v1.reduce_mean(tf.cast(correct_pred,tf.float32))
        
