# coding=utf-8
'''
This script is based on https://github.com/cjymz886/text_bert_cnn and https://github.com/google-research/bert.
'''
import collections
import codecs
import os
import sys
import time
from sklearn import metrics
import tensorflow as tf
import numpy as np

from encodetoken import tokenization
from text_model import *
import os
tf.compat.v1.disable_eager_execution()

class TextConfig():
    seq_length = 128  # max length of sentence
    num_labels = 2  # number of labels

    num_filters = 128  # number of convolution kernel
   
    filter_sizess = [128,128,128]  
    hidden_dim = 128 # number of fully_connected layer units 128

    keep_prob = 0.5 # droppout0.4
    lr = 5e-5  # learning rate 5e-5
    lr_decay = 0.9  # learning rate decay 0.9
    clip = 5.0  # gradient clipping threshold

    is_training = True  # is _training
    use_one_hot_embeddings = False  # use_one_hot_embeddings

    num_epochs = 64  # epochs
    batch_size = 32  # batch_size 
    print_per_batch = 10 # print result200
    require_improvement = 1000  # stop training if no inporement over 1000 global_step

    output_dir = './result'
    data_dir = './corpus/TCM/'  # the path of input_data file
    training_data = data_dir + 'SD_fold5_training.txt'
    dev_data = data_dir + 'SD_fold5_test.txt'
    test_data = data_dir + 'SD_fold5_test.txt'

    BERT_BASE_DIR = './vocab/'
    vocab_file = BERT_BASE_DIR+'vocab.txt'  # the path of vocab file


class Processor(object):
    """Processor for the TCM corpus."""
    def get_train_examples(self, training_data):
        return self._create_examples(
            self._read_file(training_data), "train")

    def get_dev_examples(self, dev_data):
        return self._create_examples(
            self._read_file(dev_data), "dev")

    def get_test_examples(self, test_data):
        return self._create_examples(
            self._read_file(test_data), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    """read file"""
    def _read_file(self, input_file):
        with codecs.open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                try:
                    line = line[:-1].split('\t')
                    lines.append(line)
                except:
                    pass
            # np.random.shuffle(lines)
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 4:
                print(line)
            guid = "%s-%s-%s" % (set_type,
                                 tokenization.convert_to_unicode(line[0]), str(i))
            target = tokenization.convert_to_unicode(line[1])
            text = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[3])
            examples.append(
                InputExample(guid=guid, text_a=target, text_b=text, label=label))
            # print(guid, target, text, label)
            # exit()
        return examples


def evaluate(sess, dev_data):
    '''Calculate the average loss and accuracy of validation/test data in batch form. '''
    data_len = 0
    total_loss = 0.0
    total_acc = 0.0
    for batch_ids, batch_mask, batch_segment, batch_label in batch_iter(dev_data, config.batch_size):
        batch_len = len(batch_ids)
        data_len += batch_len
        feed_dict = feed_data(batch_ids, batch_mask,
                              batch_segment, batch_label, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss/data_len, total_acc/data_len


def feed_data(batch_ids, batch_mask, batch_segment, batch_label, keep_prob):
    '''Data for text_model construction. '''
    feed_dict = {
        model.input_ids: np.array(batch_ids),
        model.input_mask: np.array(batch_mask),
        model.segment_ids: np.array(batch_segment),
        model.labels: np.array(batch_label),
        model.keep_prob: keep_prob
    }
    return feed_dict


def train():
    '''Train the text_bert_cnn model. '''
    tensorboard_dir = os.path.join(config.output_dir, "tensorboard/TCM")
    save_dir = os.path.join(config.output_dir, "checkpoints/TCM")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    start_time = time.time()

    tf.compat.v1.logging.info("*****************Loading training data*****************")
    train_examples = Processor().get_train_examples(config.training_data)
    trian_data = convert_examples_to_features(
        train_examples, label_list, config.seq_length, tokenizer)

    tf.compat.v1.logging.info("*****************Loading dev data*****************")
    dev_examples = Processor().get_dev_examples(config.dev_data)
    dev_data = convert_examples_to_features(
        dev_examples, label_list, config.seq_length, tokenizer)

    tf.compat.v1.logging.info("Time cost: %.3f seconds...\n" %
                    (time.time() - start_time))

    tf.compat.v1.logging.info("Building session and restore bert_model...\n")
    session = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    session.run(tf.compat.v1.global_variables_initializer())

    tf.compat.v1.summary.scalar("loss", model.loss)
    tf.compat.v1.summary.scalar("accuracy", model.acc)
    merged_summary = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    tf.compat.v1.logging.info('Training and evaluating...\n')
    best_acc = 0
    last_improved = 0  # record global_step at best_val_accuracy
    flag = False
    for epoch in range(config.num_epochs):
        batch_train = batch_iter(trian_data, config.batch_size)
        start = time.time()
        tf.compat.v1.logging.info('Epoch:%d' % (epoch + 1))
        for batch_ids, batch_mask, batch_segment, batch_label in batch_train:
            feed_dicts = feed_data(batch_ids, batch_mask,
                                  batch_segment, batch_label, config.keep_prob)
           
            _, global_step, train_summaries, train_loss, train_accuracy = session.run(
               [model.optim, model.global_step, merged_summary, model.loss, model.acc], feed_dict=feed_dicts)
           
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, dev_data)
                merged_acc = (train_accuracy+val_accuracy)/2
                if merged_acc > best_acc:
                    saver.save(session, save_path)
                    best_acc = merged_acc
                    last_improved = global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                tf.compat.v1.logging.info("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}".format(
                    global_step, train_loss, train_accuracy, val_loss, val_accuracy, (end - start) / config.print_per_batch, improved_str))
                start = time.time()

            if global_step - last_improved > config.require_improvement:
                tf.compat.v1.logging.info(
                    "No optimization over 1500 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.lr *= config.lr_decay


def test():
    '''testing'''
    save_dir = os.path.join(config.output_dir, "checkpoints/TCM")
    save_path = os.path.join(save_dir, 'best_validation')

    if not os.path.exists(save_dir):
        tf.compat.v1.logging.info("maybe you don't train")
        exit()

    tf.compat.v1.logging.info("*****************Loading testing data*****************")
    test_examples = Processor().get_test_examples(config.test_data)
    test_data = convert_examples_to_features(
        test_examples, label_list, config.seq_length, tokenizer)

    input_ids, input_mask, segment_ids = [], [], []

    for features in test_data:
        input_ids.append(features['input_ids'])
        input_mask.append(features['input_mask'])
        segment_ids.append(features['segment_ids'])

    config.is_training = False
    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(session, save_path=save_path)

    tf.compat.v1.logging.info('Testing...')
    test_loss, test_accuracy = evaluate(session, test_data)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    tf.compat.v1.logging.info(msg.format(test_loss, test_accuracy))
    a=model.y_pred_cls
    batch_size = config.batch_size
    data_len = len(test_data)
    num_batch = int((data_len-1)/batch_size)+1
    y_test_cls = [features['label_ids'] for features in test_data]
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)

    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, data_len)
        feed_dict = {
            model.input_ids: np.array(input_ids[start_id:end_id]),
            model.input_mask: np.array(input_mask[start_id:end_id]),
            model.segment_ids: np.array(segment_ids[start_id:end_id]),
            model.keep_prob: 1.0,
        }
        y_pred_cls[start_id:end_id] = session.run(
            model.y_pred_cls, feed_dict=feed_dict)

    # evaluate
    tf.compat.v1.logging.info("Precision, Recall and F1-Score...")
    tf.compat.v1.logging.info(metrics.classification_report(
        y_test_cls, y_pred_cls, target_names=label_list))

    tf.compat.v1.logging.info("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    tf.compat.v1.logging.info(cm)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_TCM.py [train / test]""")

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    config = TextConfig()
    label_list = Processor().get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=config.vocab_file, do_lower_case=False)
    model = TextMLP(config)
   
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()
    else:
        exit()
