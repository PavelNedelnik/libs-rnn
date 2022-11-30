import warnings
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

def match_wavelengths(left, right, left_calibration, right_calibration):
  """
  Resample two spectral datasets to have common calibration.
  [left1(left2,right1)left3] or [left1(left2,right1]right2)
  _____________
  Returns new left, new right, new calibration
  """
  if left_calibration[0] > right_calibration[0]:
    right, left, calibration = match_wavelengths(right, left, right_calibration, left_calibration)
    return left, right, calibration

  # find the calibration for each part
  # left from the intersection
  left1 = left_calibration[left_calibration < right_calibration[0]]
  # intersection
  left2 = left_calibration[(left_calibration >= right_calibration[0]) & (left_calibration <= right_calibration[-1])]
  # right from the intersection
  left3 = left_calibration[left_calibration > right_calibration[-1]]

  # intersection
  right1 = right_calibration[(right_calibration >= left_calibration[0]) & (right_calibration <= left_calibration[-1])]
  # right from the intersection
  right2 = right_calibration[right_calibration > left_calibration[-1]]

  # combine the calibrations using the finer one for the intersection
  if np.mean(np.diff(left2)) < np.mean(np.diff(right1)):
    calibration = np.hstack((left1, left2, right2, left3))
  else:
    calibration = np.hstack((left1, right1, right2, left3))

  # resample the spectra
  left = np.apply_along_axis(lambda x: np.interp(calibration, left_calibration, x, left=0., right=0.), 1, left)
  right = np.apply_along_axis(lambda x: np.interp(calibration, right_calibration, x, left=0., right=0.), 1, right)
  return left, right, calibration


class LabelCropp(BaseEstimator, TransformerMixin):
  """
  TODO add nonnumeric labels
  """

  def __init__(self, label_from=None, label_to=None, labels=None):
    # TODO default values!!!
    self.label_from = label_from
    self.label_to = label_to
    self.labels = labels


  def fit(self, X, y=None):
    X = check_array(X)
    self.n_features_in_ = X.shape[1]
    return self


  def transform(self, X, y=None):
    check_is_fitted(self)
    X = check_array(X)
    if X.shape[1] != self.n_features_in_:
      raise ValueError('Fit ({}) and Transform ({}) data does not match shape!'.format(self.n_features_in_, X.shape[1]))
    labels = self.labels if self.labels is not None else np.arange(self.n_features_in_)
    label_from = self.label_from if self.label_from is not None else labels[0]
    label_to = self.label_to if self.label_to is not None else labels[-1]
      
    if label_from > labels.max() or label_to < labels.min():
      warnings.warn('Labels out of range! Skipping!')
      return X

    idx_from, idx_to = np.argmax(labels >= label_from),  len(labels) - np.argmax((labels <= label_to)[::-1]) - 1
    return X[:, idx_from: idx_to]