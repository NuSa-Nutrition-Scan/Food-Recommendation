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
  """Model for encoding foods."""

  def __init__(self, layer_sizes, popular_weight=1, retrieval_weight=1):
    """Model for encoding foods.

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

weights_model_filepath = './saved_model/model_weight'
model_2 = FoodlensModel(layer_sizes=None,popular_weight=1, retrieval_weight=1)
model_2.load_weights(weights_model_filepath).expect_partial()

# Fungsi ini membutuhkan:
# food_data_raw -> data makanan di database
# input dict    -> input dict mengenai data user (dalam bentuk tf.constant)
# output type   -> "print", "dataframe", "dict"
# model_recom   -> model recommendation system
# top_n         -> seberapa banyak rekomendasi makanan yang dihasilkan

def predict_food(food_data_raw,input_dict,output_type, model_recom ,top_n=3):
    USER_FEATURE_NUM = ['Age', 'Body_Weight', 'Body_Height','Cal_Need']
    
    USER_FEATURE_CAT= ['sex','blood_group']
    
    food_data = food_data_raw.set_index('Food_ID').reset_index().drop(food_data_raw.columns[[0,31,32,33,34,35,36]],axis = 1).copy()
    food_data['Food_ID'] = food_data['Food_ID'].astype('str')
    
    foods = tf.data.Dataset.from_tensor_slices(dict(food_data[['Food_ID','Fast_Food','Sumber','Tipe',
                                                               'Jenis_Olahan','Mentah / Olahan','Kelompok Makanan','Air (g)', 'Energi (Kal)','Protein (g)',
                                                               'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)',
                                                               'Abu (g)','Kalsium (Ca) (mg)', 'Fosfor (P) (mg)', 'Besi (Fe) (mg)',
                                                               'Natrium (Na) (mg)', 'Kalium (Ka) (mg)', 'Tembaga (Cu) (mg)',
                                                               'Seng (Zn) (mg)', 'Retinol (vit. A) (mcg)', 'β-karoten (mcg)',
                                                               'Karoten total (mcg)', 'Thiamin (vit. B1) (mg)',
                                                               'Riboflavin (vit. B2) (mg)', 'Niasin (mg)', 'Vitamin C (mg)', 'BDD (%)']]))

    # Create a model that takes in raw query features, and
    brute_force = tfrs.layers.factorized_top_k.BruteForce(model_recom.query_model, k = top_n)
    
    # recommends foods out of the entire foods dataset.
    brute_force.index_from_dataset(foods.apply(tf.data.experimental.dense_to_ragged_batch(151)).map(model_recom.candidate_model))

    recommended_food = brute_force({
    "User_ID": tf.constant([input_dict['User_ID'].numpy()[0].decode("utf-8")]),
    **{k: tf.constant([input_dict[k].numpy()[0]]) for k in USER_FEATURE_NUM+['sex']}
    })
    
    if output_type=="print":
        print('Top {} recommendations for user {}:\n'.format(top_n, input_dict['User_ID']))
        for i, food_id in enumerate(recommended_food[1].numpy()[0,:top_n]):
            if list(food_data_raw[food_data_raw["No."]==food_id+1]["Food_ID"])==[]:
                continue
            print('{}. {} : {}'.format(i+1, list(food_data_raw[food_data_raw["No."]==food_id+1]["Food_ID"])[0], list(food_data_raw[food_data_raw["No."]==food_id+1]["Nama Bahan Makanan"])[0]))
    
    if output_type=="dataframe":
        df_output = pd.DataFrame()

        df_output['index_number'] = list(range(1,top_n+1))
        df_output['list_food_id'] = [list(food_data_raw[food_data_raw["No."]==index+1]["Food_ID"])[0] for index in recommended_food[1].numpy()[0,:top_n]]
        df_output['list_food_name'] = [list(food_data_raw[food_data_raw["No."]==index+1]["Nama Bahan Makanan"])[0] for index in recommended_food[1].numpy()[0,:top_n]]
        return df_output

    if output_type=="dict":
        df_output = pd.DataFrame()
        
        df_output['index_number'] = list(range(1,top_n+1))
        df_output['list_food_id'] = [list(food_data_raw[food_data_raw["No."]==index+1]["Food_ID"])[0] for index in recommended_food[1].numpy()[0,:top_n]]
        df_output['list_food_name'] = [list(food_data_raw[food_data_raw["No."]==index+1]["Nama Bahan Makanan"])[0] for index in recommended_food[1].numpy()[0,:top_n]]
        return df_output.to_dict('dict')


# Fungsi ini membutuhkan:
# food_data_raw   -> data makanan di database
# dict_new_user   -> input dict mengenai data user (dalam bentuk tf.constant)
# dict_food_data  -> input dict mengenai data food (dalam bentuk tf.constant)
# model_recom     -> model recommendation system

def predict_popular(food_data_raw, dict_new_user,dict_food_data,model_recom):

    food_data = food_data_raw.set_index('Food_ID').reset_index().drop(food_data_raw.columns[[0,31,32,33,34,35,36]],axis = 1).copy()
    food_data['Food_ID'] = food_data['Food_ID'].astype('str')

    foods = tf.data.Dataset.from_tensor_slices(dict(food_data[['Food_ID','Fast_Food','Sumber','Tipe',
                                                               'Jenis_Olahan','Mentah / Olahan','Kelompok Makanan','Air (g)', 'Energi (Kal)','Protein (g)',
                                                               'Lemak (g)', 'Karbohidrat (g)', 'Serat (g)',
                                                               'Abu (g)','Kalsium (Ca) (mg)', 'Fosfor (P) (mg)', 'Besi (Fe) (mg)',
                                                               'Natrium (Na) (mg)', 'Kalium (Ka) (mg)', 'Tembaga (Cu) (mg)',
                                                               'Seng (Zn) (mg)', 'Retinol (vit. A) (mcg)', 'β-karoten (mcg)',
                                                               'Karoten total (mcg)', 'Thiamin (vit. B1) (mg)',
                                                               'Riboflavin (vit. B2) (mg)', 'Niasin (mg)', 'Vitamin C (mg)', 'BDD (%)']]))

    input_dict_total = dict(dict_new_user)
    input_dict_total.update(dict_food_data)
    input_dict_total= {k: tf.constant([input_dict_total[k]]) for k in input_dict_total}
    predicted_popular = model_recom.predict(input_dict_total)
    print("Predicted popular for {} or {}: {}".format(input_dict_total['Food_ID'][0].numpy().decode("utf-8"), list(food_data[food_data['Food_ID']==input_dict_total['Food_ID'][0].numpy().decode("utf-8")]['Nama Bahan Makanan'])[0],predicted_popular[0,0]))

# Fungsi ini membutuhkan:
# food_data_raw -> data makanan di database
# list_recom_food -> output from predict_food function
# gender
# pred_cal -> user prediction calorie or need
# amount of eat

def top_nutrition(food_data_raw,user_id, list_recom_food, gender, pred_cal=None, amount_of_eat=3):
    data_nut_food = food_data_raw[food_data_raw["Nama Bahan Makanan"].isin(list_recom_food)][["Nama Bahan Makanan","Energi (Kal)","Protein (g)","Lemak (g)","Karbohidrat (g)"]]
    data_nut_cal = data_nut_food[["Nama Bahan Makanan","Energi (Kal)"]]
    data_nut_pro = data_nut_food[["Nama Bahan Makanan","Protein (g)"]]
    data_nut_fat = data_nut_food[["Nama Bahan Makanan","Lemak (g)"]]
    data_nut_carb = data_nut_food[["Nama Bahan Makanan","Karbohidrat (g)"]]

    if pred_cal is None:
        if gender=="M":
            pred_cal=2500
        else:
            pred_cal=2000
    
    if gender=="M":
        protein_need = 55
        carb_need = 275
        fat_need = 67
    else:
        protein_need = 45
        carb_need = 275
        fat_need = 67

    if amount_of_eat == 2:
        comb_2 = combinations(list_recom_food, 2)
        list_cal = [np.sum(np.power(np.subtract(np.multiply(nut,4),pred_cal),2)) for nut in [[list(data_nut_cal[data_nut_cal["Nama Bahan Makanan"]==str(food)]["Energi (Kal)"])[0] for food in comb] for comb in list(comb_2)]]

        comb_2 = combinations(list_recom_food, 2)
        list_pro = [np.sum(np.power(np.subtract(np.multiply(nut,4),protein_need),2)) for nut in [[list(data_nut_pro[data_nut_pro["Nama Bahan Makanan"]==str(food)]["Protein (g)"])[0] for food in comb] for comb in list(comb_2)]]

        comb_2 = combinations(list_recom_food, 2)
        list_fat = [np.sum(np.power(np.subtract(np.multiply(nut,4),protein_need),2)) for nut in [[list(data_nut_fat[data_nut_fat["Nama Bahan Makanan"]==str(food)]["Lemak (g)"])[0] for food in comb] for comb in list(comb_2)]]

        comb_2 = combinations(list_recom_food, 2)
        list_carb = [np.sum(np.power(np.subtract(np.multiply(nut,4),protein_need),2)) for nut in [[list(data_nut_carb[data_nut_carb["Nama Bahan Makanan"]==str(food)]["Karbohidrat (g)"])[0] for food in comb] for comb in list(comb_2)]]

        total_list = [sum(x) for x in zip(list_cal,list_pro,list_fat,list_carb)]

        comb_2 = combinations(list_recom_food, 2)
        list_mse = {comb:total_list[i] for i, comb in enumerate(comb_2)}
        list_mse_sorted = sorted(list_mse.items(), key=lambda x:x[1])
        
        return pd.DataFrame([list(x) for x in list_mse_sorted]).to_dict('dict')[0]

    elif amount_of_eat == 4:
        comb_4 = combinations(list_recom_food, 4)
        list_cal = [np.sum(np.power(np.subtract(np.multiply(nut,2),pred_cal),2)) for nut in [[list(data_nut_cal[data_nut_cal["Nama Bahan Makanan"]==str(food)]["Energi (Kal)"])[0] for food in comb] for comb in list(comb_4)]]
        
        comb_4 = combinations(list_recom_food, 4)
        list_pro = [np.sum(np.power(np.subtract(np.multiply(nut,2),protein_need),2)) for nut in [[list(data_nut_pro[data_nut_pro["Nama Bahan Makanan"]==str(food)]["Protein (g)"])[0] for food in comb] for comb in list(comb_4)]]

        comb_4 = combinations(list_recom_food, 4)
        list_fat = [np.sum(np.power(np.subtract(np.multiply(nut,2),protein_need),2)) for nut in [[list(data_nut_fat[data_nut_fat["Nama Bahan Makanan"]==str(food)]["Lemak (g)"])[0] for food in comb] for comb in list(comb_4)]]

        comb_4 = combinations(list_recom_food, 4)
        list_carb = [np.sum(np.power(np.subtract(np.multiply(nut,2),protein_need),2)) for nut in [[list(data_nut_carb[data_nut_carb["Nama Bahan Makanan"]==str(food)]["Karbohidrat (g)"])[0] for food in comb] for comb in list(comb_4)]]

        total_list = [sum(x) for x in zip(list_cal,list_pro,list_fat,list_carb)]

        comb_4 = combinations(list_recom_food, 4)
        list_mse = {comb:total_list[i] for i, comb in enumerate(comb_4)}
        list_mse_sorted = sorted(list_mse.items(), key=lambda x:x[1])
        
        return pd.DataFrame([list(x) for x in list_mse_sorted]).to_dict('dict')[0]

    else:
        comb_3 = combinations(list_recom_food, 3)
        list_cal = [np.sum(np.power(np.subtract(np.multiply(nut,2.7),pred_cal),2)) for nut in [[list(data_nut_cal[data_nut_cal["Nama Bahan Makanan"]==str(food)]["Energi (Kal)"])[0] for food in comb] for comb in list(comb_3)]]

        comb_3 = combinations(list_recom_food, 3)
        list_pro = [np.sum(np.power(np.subtract(np.multiply(nut,2.7),protein_need),2)) for nut in [[list(data_nut_pro[data_nut_pro["Nama Bahan Makanan"]==str(food)]["Protein (g)"])[0] for food in comb] for comb in list(comb_3)]]

        comb_3 = combinations(list_recom_food, 3)
        list_fat = [np.sum(np.power(np.subtract(np.multiply(nut,2.7),protein_need),2)) for nut in [[list(data_nut_fat[data_nut_fat["Nama Bahan Makanan"]==str(food)]["Lemak (g)"])[0] for food in comb] for comb in list(comb_3)]]

        comb_3 = combinations(list_recom_food, 3)
        list_carb = [np.sum(np.power(np.subtract(np.multiply(nut,2.7),protein_need),2)) for nut in [[list(data_nut_carb[data_nut_carb["Nama Bahan Makanan"]==str(food)]["Karbohidrat (g)"])[0] for food in comb] for comb in list(comb_3)]]

        total_list = [sum(x) for x in zip(list_cal,list_pro,list_fat,list_carb)]

        comb_3 = combinations(list_recom_food, 3)
        list_mse = {comb:total_list[i] for i, comb in enumerate(comb_3)}
        list_mse_sorted = sorted(list_mse.items(), key=lambda x:x[1])
        
        return pd.DataFrame([list(x) for x in list_mse_sorted]).to_dict('dict')[0]

import gradio as gr

with gr.Blocks() as demo:
    User_ID = gr.Text(label="User_ID",placeholder="UNT001")
    Age = gr.Number(label="Age")
    Body_Weight = gr.Number(label="Body_Weight")
    Body_Height = gr.Number(label="Body_Height")
    Cal_Need = gr.Number(label="Cal_Need")
    Gender = gr.Text(label="Gender",placeholder="M or F")
    Amount_Of_Eat = gr.Number(label="Amount_Of_Eat (2 or 3 or 4)")

    with gr.Row():
        recom_btn = gr.Button("Generate Recommender Food and Top Nutrition")

    recom_out = gr.Dataframe(row_count = (3, "dynamic"), col_count=(3, "fixed"), label="Food Recommendations", headers=["Index","Food ID","Food Names"])
    topnut_out = gr.Dataframe(row_count = (3, "dynamic"), col_count=(4, "dynamic"), label="Top Pair Nutritions", headers=["Breakfast","Lunch","Dinner","Snacks"])
    
    def recom_food_gradio(User_ID,Age,Body_Weight,Body_Height,Cal_Need,Gender,Amount_Of_Eat):
        list_food_name = predict_food(food_data_raw=food_data_raw,
             input_dict={
                 "User_ID": tf.constant([User_ID]),
                 "Age":tf.constant([Age]),
                 "Body_Weight":tf.constant([Body_Weight]),
                 "Body_Height":tf.constant([Body_Height]),
                 "Cal_Need":tf.constant([Cal_Need]),
                 "sex":tf.constant([Gender])
             },
             output_type = "dict",
             model_recom = model_2,
             top_n=15)

        list_food_df = pd.DataFrame(list_food_name)
        list_food_df.columns = ["Index","Food ID","Food Names"]

        list_food = list(pd.DataFrame(list_food_name)['list_food_name'])

        top_nutri_grad = top_nutrition(food_data_raw = food_data_raw,
                                       user_id = User_ID,
                                       list_recom_food = list_food,
                                       gender = Gender,
                                       pred_cal = Cal_Need,
                                       amount_of_eat=Amount_Of_Eat)

        top_nutri_df = pd.DataFrame(top_nutri_grad).T
        
        if Amount_Of_Eat==2:
            top_nutri_df.columns = ["Lunch", "Dinner"]

        elif Amount_Of_Eat==4:
            top_nutri_df.columns = ["Breakfast","Lunch", "Dinner","Snacks"]

        else:
            top_nutri_df.columns = ["Breakfast","Lunch", "Dinner"]

        return list_food_df, top_nutri_df

    recom_btn.click(recom_food_gradio, inputs=[User_ID, Age, Body_Weight, Body_Height, Cal_Need, Gender, Amount_Of_Eat], outputs=[recom_out,topnut_out])

demo.launch(enable_queue=True)