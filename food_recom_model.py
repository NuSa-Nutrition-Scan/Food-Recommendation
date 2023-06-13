import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_recommenders as tfrs
from typing import Dict, Text
from itertools import combinations

user_data_raw = pd.read_pickle("./user_data.pkl")
food_data_raw = pd.read_pickle("./food_raw.pkl")
food_popularity_raw = pd.read_pickle("./food_popularity.pkl")

food_data = food_data_raw.set_index('Food_ID').reset_index().drop(food_data_raw.columns[[0,31,32,33,34,35,36]],axis = 1).copy()
food_data['Food_ID'] = food_data['Food_ID'].astype('str')

populars = tf.data.Dataset.from_tensor_slices(dict(food_popularity_raw[['User_ID', 'Food_ID', 'value',
'Age', 'Body_Weight', 'Body_Height','Cal_Need','sex','blood_group','Fast_Food','Sumber','Tipe',
'Jenis_Olahan','Mentah / Olahan','Kelompok Makanan','Air (g)', 'Energi (Kal)','Protein (g)',
'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)',
'Abu (g)','Kalsium (Ca) (mg)', 'Fosfor (P) (mg)', 'Besi (Fe) (mg)',
'Natrium (Na) (mg)', 'Kalium (Ka) (mg)', 'Tembaga (Cu) (mg)',
'Seng (Zn) (mg)', 'Retinol (vit. A) (mcg)', 'β-karoten (mcg)',
'Karoten total (mcg)', 'Thiamin (vit. B1) (mg)',
'Riboflavin (vit. B2) (mg)', 'Niasin (mg)', 'Vitamin C (mg)', 'BDD (%)']]))

foods = tf.data.Dataset.from_tensor_slices(dict(food_data[['Food_ID','Fast_Food','Sumber','Tipe',
'Jenis_Olahan','Mentah / Olahan','Kelompok Makanan','Air (g)', 'Energi (Kal)','Protein (g)',
'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)',
'Abu (g)','Kalsium (Ca) (mg)', 'Fosfor (P) (mg)', 'Besi (Fe) (mg)',
'Natrium (Na) (mg)', 'Kalium (Ka) (mg)', 'Tembaga (Cu) (mg)',
'Seng (Zn) (mg)', 'Retinol (vit. A) (mcg)', 'β-karoten (mcg)',
'Karoten total (mcg)', 'Thiamin (vit. B1) (mg)',
'Riboflavin (vit. B2) (mg)', 'Niasin (mg)', 'Vitamin C (mg)', 'BDD (%)']]))

food_names = foods.batch(100).map(tf.autograph.experimental.do_not_convert(lambda x: x["Food_ID"]))
user_ids = populars.batch(100).map(tf.autograph.experimental.do_not_convert(lambda x: x["User_ID"]))
unique_food_names = np.unique(np.concatenate(list(food_names)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

USER_FEATURE_NUM = ['Age', 'Body_Weight', 'Body_Height','Cal_Need']

USER_FEATURE_CAT= ['sex','blood_group']

FOOD_FEATURE_NUM = ['Air (g)', 'Energi (Kal)','Protein (g)', 'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)',
'Abu (g)','Kalsium (Ca) (mg)', 'Fosfor (P) (mg)', 'Besi (Fe) (mg)',
'Natrium (Na) (mg)', 'Kalium (Ka) (mg)', 'Tembaga (Cu) (mg)',
'Seng (Zn) (mg)', 'Retinol (vit. A) (mcg)', 'β-karoten (mcg)',
'Karoten total (mcg)', 'Thiamin (vit. B1) (mg)',
'Riboflavin (vit. B2) (mg)', 'Niasin (mg)', 'Vitamin C (mg)', 'BDD (%)']

FOOD_FEATURE_CAT = ['Fast_Food', 'Tipe','Sumber','Jenis_Olahan',
'Mentah / Olahan','Kelompok Makanan']

class UserModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, 64),
    ])

    self.additional_feature = {}
    self.normalized = {}
    self.categorized = {}

    for feature in USER_FEATURE_NUM:
        self.normalized[feature] = tf.keras.layers.Normalization(axis=None)
        self.normalized[feature].adapt(populars.map(lambda x: x[feature]))
        self.additional_feature[feature] = tf.keras.Sequential([self.normalized[feature],tf.keras.layers.Reshape([1])])

    self.categorized['sex'] = tf.keras.layers.StringLookup(vocabulary=np.unique(np.concatenate(list(populars.batch(100).map(lambda x: x["sex"])))), mask_token=None)
    self.additional_feature['sex'] = tf.keras.Sequential([self.categorized['sex'],tf.keras.layers.Embedding(3, 8)])

  def call(self, inputs):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    
    return tf.concat(
        [self.user_embedding(inputs["User_ID"])]+
        [self.additional_feature[k](inputs[k]) for k in self.additional_feature],
        axis=1)
  

class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""

  def __init__(self, layer_sizes, popular_weight=1, retrieval_weight=1):
    """Model for encoding user queries.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    # We first use the user model for generating embeddings.
    self.user_embedding_model = UserModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the linear activation
    self.dense_layers.add(tf.keras.layers.Dense(128))

  def call(self, inputs):
    feature_embedding = self.user_embedding_model(inputs)
    return self.dense_layers(feature_embedding)

class FoodModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.food_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
          vocabulary=unique_food_names,mask_token=None),
      tf.keras.layers.Embedding(len(unique_food_names) + 1, 64)
    ])

    self.additional_feature = {}
    self.normalized={}
    self.categorized={}

    for feature in FOOD_FEATURE_NUM:
        self.normalized[feature] = tf.keras.layers.Normalization(axis=None)
        self.normalized[feature].adapt(populars.map(lambda x: x[feature]))
        self.additional_feature[feature] = tf.keras.Sequential([self.normalized[feature],tf.keras.layers.Reshape([1])])

    for feature in FOOD_FEATURE_CAT:
        self.categorized[feature] = tf.keras.layers.StringLookup(vocabulary=np.unique(np.concatenate(list(foods.batch(100).map(lambda x: x[feature])))),mask_token=None)
        self.additional_feature[feature] = tf.keras.Sequential([self.categorized[feature],tf.keras.layers.Embedding(len(np.unique(np.concatenate(list(foods.batch(151).map(lambda x: x[feature])))))+1, 8)])

  def call(self, inputs):
      return tf.concat(
          [self.food_embedding(inputs["Food_ID"])]+
          [self.additional_feature[k](inputs[k]) for k in self.additional_feature],
          axis=1)

class CandidateModel(tf.keras.Model):
  """Model for encoding movies."""

  def __init__(self, layer_sizes, popular_weight=1, retrieval_weight=1):
    """Model for encoding movies.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    self.food_embedding_model = FoodModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the linear activation.
    self.dense_layers.add(tf.keras.layers.Dense(128))

  def call(self, inputs):
    feature_embedding = self.food_embedding_model(inputs)
    return self.dense_layers(feature_embedding)


class FoodlensModel(tfrs.models.Model):

  def __init__(self, layer_sizes, popular_weight=1, retrieval_weight=1):
    super().__init__()
    self.query_model = QueryModel(layer_sizes)
    self.candidate_model = CandidateModel(layer_sizes)

    self.popular_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.popular_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=foods.apply(tf.data.experimental.dense_to_ragged_batch(151)).map(self.candidate_model)
        )
    )

    # The loss weights.
    self.popular_weight = popular_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor], training=True) -> tf.Tensor:
      
      query_embeddings = self.query_model({"User_ID": features["User_ID"],
      **{k: features[k] for k in USER_FEATURE_NUM+['sex']}
      })
      food_embeddings = self.candidate_model({"Food_ID": features["Food_ID"],
      **{k: features[k] for k in FOOD_FEATURE_NUM+FOOD_FEATURE_CAT}
      })

      output_dot = tf.concat([query_embeddings, food_embeddings],axis=1)

      return self.popular_model(output_dot)

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "User_ID": features["User_ID"],
        **{k: features[k] for k in USER_FEATURE_NUM+['sex']}
        })
    food_embeddings = self.candidate_model({
        "Food_ID": features["Food_ID"],
        **{k: features[k] for k in FOOD_FEATURE_NUM + FOOD_FEATURE_CAT}
        })

    populars_value = features.pop("value")

    popular_predictions = self(features)

    # We compute the loss for each task.
    popular_loss = self.popular_task(
        labels=populars_value,
        predictions=popular_predictions)

    retrieval_loss = self.retrieval_task(query_embeddings, food_embeddings, compute_metrics=not training)

    return (self.popular_weight * popular_loss + self.retrieval_weight * retrieval_loss)