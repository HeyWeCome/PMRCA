import logging
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import Constants
from utils.Metrics import Metrics
from utils.Optim import ScheduledOptim

class BaseRunner(object):
    def __init__(self, args):
        self.patience = 10

    def run(self, model, train_data, valid_data, test_data, args):
        loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=Constants.PAD)
        params = model.parameters()
        adam = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-09)
        optimizer = ScheduledOptim(adam, args.d_model, args.n_warmup_steps)

        model = model.to(args.device)
        loss_func = loss_func.to(args.device)

        validation_history = 0.0
        best_scores = {}
        epochs_without_improvement = 0

        for epoch_i in range(args.epoch):
            logging.info(f'\n[ Epoch {epoch_i} ]')
            start = time.time()

            train_loss, train_accu = self.train_epoch(model, train_data, loss_func, optimizer)
            logging.info(
                f'  \n- (Training)   '
                f'Loss: {train_loss:8.5f} | '
                f'Accuracy: {100 * train_accu:3.3f}% | '
                f'Elapsed: {(time.time() - start) / 60:3.3f} min')

            start = time.time()
            validation_scores = self.test_epoch(model, valid_data)
            logging.info('\n  - (Validation)')

            logging.info(f"{'Metric':<15} {'Score':<20}")
            logging.info('-' * 35)
            for metric, score in validation_scores.items():
                logging.info(f"{metric:<15} {score:<20.6f}")

            logging.info(f'Validation use time: {(time.time() - start) / 60:.3f} min')

            current_validation_score = sum(validation_scores.values())
            if current_validation_score > validation_history:
                logging.info(f"\nBest Validation at Epoch: {epoch_i}, model has been saved.")
                validation_history = current_validation_score
                best_scores = validation_scores
                torch.save(model.state_dict(), args.model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logging.info(f"No improvement in validation score for {epochs_without_improvement} epochs.")

            if epochs_without_improvement >= self.patience:
                logging.info(f"\nEarly stopping triggered after {epoch_i + 1} epochs.")
                break

        logging.info('\n  - (Final Test)')
        test_scores = self.test_epoch(model, test_data)

        logging.info(f"{'Metric':<15} {'Score':<20}")
        logging.info('-' * 35)
        for metric, score in test_scores.items():
            logging.info(f"{metric:<15} {score:<20.6f}")

        logging.info("\n- (Finished!!) \nBest validation scores:")
        logging.info(f"{'Metric':<15} {'Score':<20}")
        logging.info('-' * 35)
        for metric, score in best_scores.items():
            logging.info(f"{metric:<15} {score:<20.6f}")

    def train_epoch(self, model, training_data, loss_func, optimizer):
        model.train()

        total_loss = 0.0
        n_total_users = 0.0
        n_total_correct = 0.0

        for batch in tqdm(training_data, desc="Training Epoch", ncols=100):
            history_seq, history_seq_timestamp, history_seq_idx = (item.cuda() for item in batch)

            gold = history_seq[:, 1:]

            n_users = gold.data.ne(Constants.PAD).sum().float()
            n_total_users += n_users

            optimizer.zero_grad()

            if hasattr(model, 'before_epoch'):
                model.before_epoch()

            loss, n_correct = model.get_performance(history_seq,
                                                    history_seq_timestamp,
                                                    history_seq_idx,
                                                    loss_func,
                                                    gold)
            loss.backward()

            optimizer.step()
            optimizer.update_learning_rate()

            n_total_correct += n_correct
            total_loss += loss.item()

        return total_loss / n_total_users, n_total_correct / n_total_users

    def test_epoch(self, model, validation_data, k_list=[10, 20, 50, 100]):
        model.eval()

        scores = {}
        for k in k_list:
            scores['hits@' + str(k)] = 0
            scores['map@' + str(k)] = 0

        n_total_words = 0
        with torch.no_grad():
            for batch in tqdm(validation_data, desc="Testing Epoch"):
                history_seq, history_seq_timestamp, history_seq_idx = (item.cuda() for item in batch)

                gold = history_seq[:, 1:].contiguous().view(-1).detach().cpu().numpy()

                pred = model(history_seq, history_seq_timestamp, history_seq_idx)
                y_pred = pred.detach().cpu().numpy()

                metric = Metrics()
                scores_batch, scores_len = metric.compute_metric(y_pred, gold, k_list)
                n_total_words += scores_len
                for k in k_list:
                    scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                    scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores

    def get_performance(self, crit, pred, gold):
        loss = crit(pred, gold.contiguous().view(-1))
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
        return loss, n_correct
    