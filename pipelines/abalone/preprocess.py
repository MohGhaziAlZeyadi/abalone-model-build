import os
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

if __name__ == "__main__":
    
    
    #install needed pakages:
    #install("tensorflow==2.6.2")
    #install("pickle-mixin")
    #install("keras==2.6.0")
    #install("protobuf==3.20.")

    
    logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    print(input_data)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/raw-data.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.info("Reading downloaded data.")
    
    ####################################################
    
  
    # Read each CSV file into DataFrame
    # This creates a list of dataframes
    data= pd.read_csv(fn, encoding='latin1')
    
    data = data.drop(['Submit Date', 'Dealer', 'VIN', 'No. of Vehicles', 'Engine', 'CSN No',
           'Off Line Date', 'Analogy Car Info','Remark', 'TAC Closing Date',
          'Onsite Support', 'Support Engineer', 'Result Evaluation ', 'Registered by',
           'OpenDays', 'Close Type', 'Issue Rectified', 'First Time Fixed', 'Level',
            'Telephone', 'Province','City', 'Originator', 'Country', 'TAC and ASC information exchange', 
                 'TAC assist info', 'Corrective Action', 'Solution'], axis = 1)
    
    
    #preporssing TAC Closing Summing-up col
    data['TAC Closing Summing-up'].value_counts()
    data.rename(columns = {'TAC Closing Summing-up':'TAC_Closing_Summing_up'}, inplace = True)
    print(data['TAC_Closing_Summing_up'].isnull().sum())
    print(data.shape)
    data = data.dropna(subset=['TAC_Closing_Summing_up'])
    print(data.shape)
    #Exetract clean TAC_Closing_Summing_up Text
    Items  = list()
    for row in data.itertuples():
        temp=""
        temp = row.TAC_Closing_Summing_up.partition('¡ª¡ª')[0].strip() 
        Items.append(temp)

    data['newCol'] = Items
    data['TAC_Closing_Summing_up'] = data['newCol']
    data = data.drop(['newCol'], axis = 1)


    #####################################################
    #Convert the string labels to lists of strings
    #The initial labels are represented as raw strings. Here we make them List[str] for a more compact representation.
    #data_filtered["TAC_Closing_Summing_up"] = data_filtered["TAC_Closing_Summing_up"].str.lower()
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('replace','Replace')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace(' ','_')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('.','')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('_the_','_')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('P050462£¬Replace_BLS','Replace_BLS')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('P032200£¬Replace_CKP','Replace_CKP')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('B147201£¬Replace_speed_regulation_resistor','Replace_speed_regulation_resistor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('P032200£¬Replace_CKP','Replace_CKP')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_AC_resistor','Replace_AC_speed_regulator_resistor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_FR_Wheel_Speed_Sensor','Replace_FR_WSS')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_sunroof_frame_and_sunshade','Replace_sunroof_sunshade')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('No_Feedback_within_48h','No_Response_from_retailer_within_48_hours')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('No_Response_from_Retailer_within_48_hours','No_Response_within_48_hours')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('PTC_heater','Replace_PTC_Heater')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_RH/LH_door_weather_strip_seal','Replace_PTC_Heater')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_brake_light_switch','Replace_BLS')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_intermediate_shaft','Replace_intermediate_shaft')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_rear_left_tail_lamp','Replace_tail_lamp')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_FL_and_RL_hub_bearing','Replace_hub_bearing')
    data["TAC_Closing_Summing_up"].str.replace('Replace_FL_Door_Latch','Replace_FL_door_latch')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Intermediate_shafts_must_be_Replaced','Replace_intermediate_shaft')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_intermediate_Shaft','Replace_intermediate_shaft')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_RR_Shock_absorber','Replace_shock_absorber')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replacement_of_OBC','Replace_OBC')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('OBC','Replace_OBC')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_AVM_Module','Replace_AVM')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Sunshade_Replacement','Replace_sunshade')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_weatherstrip_seal_rear_left_and_rear_right','Replace_weatherstrip_seal')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_both_rear_wheel_speed_sensors','Replace_WSS')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_rear_both_shock_absorber','Replace_shock_absorber')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_clutch_kit_and_CSC','Replace_CSC_and_clutch')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('renew_RL_regulator_motor','Replace_window_regulator')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_rear_right_bearing_hub','Replace_hub_bearing')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('No_feedback_from_dealer','No_reply_from_dealer')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_TRANSMISSION_VALVE_BODY','Replace_valve_body')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_battery','Replace_battery')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('ptc_heater','Replace_PTC_Heater')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_driver_loor_latch','Replace_door_latch')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_driver_latch','Replace_door_latch')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACED_STEERING_COLUMN_ASSEMBLY','Replace_steering_column')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_EPS','Replace_EPS_column')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('add_a_new_case_when_stock_are_available','add_a_new_case_when_stock_arrives')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Replace_OBC','Replace_OBC')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Steering_column','Replace_steering_column')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_steering_column','Replace_steering_column')




    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_CR','Replace_color_radio')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_EPS_column','Replace_EPS_module')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('No_Response_from_retailer_within_48_hours','No_Response_from_Retailer_within_48_hours')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_rear_camera','Replace_camera')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_front_camera','Replace_camera')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_three_catalytic_coverter','Replace_catalytic_converter')

    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_INSTRUMENT_PACK_(IPK)','Replace_IPK')

    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_IPK_Assy','Replace_IPK')


    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_AC_speed_regulator_resistance','Replace_AC_speed_regulator_resistor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_both_bearing_hub','Replace_hub_bearing')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_steering_gear_and_refill_fliud','Replace_steering_gear_and_refill')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_steering_gear_and_refill', 'Replace_steering_gear')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_steering_Gear_assy', 'Replace_steering_gear')


    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_FICM_module','Replace_FICM')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_high_pressure_fuel_pipe','Replace_fuel_pipe')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Sunroof_motor','Replace_sunroof_motor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_FL_hub_bearing','Replace_hub_bearing')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_complete_cylinder_head_assembly','Replace_cylinder_head')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_aircon_compressor','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Please_Replace_ECM','Replace_ECM')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_top_mounting_bearing','Replace_top_mounting_plate')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_cable','Replace_charging_cable')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_valve_body_unit_and_harness','Replace_valve_body_and_harness')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('update_FICM','Update_FICM')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_rear_right_bearing_hub','Replace_hub_bearing')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_left_DRL_headlamp','Replace_headlamp')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_DRL_headlamp_both_side','Replace_headlamp')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Renew_FL_headlamp','Replace_headlamp')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Please_Replace_headlamps','Replace_headlamp')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Please_procede_with_Replacemente_of_EPS','Replace_EPS_module')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Fuse_box','Replace_fuse_box')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_LH_tail_lamp','Replace_tail_lamp')

    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_steering_column_assembly','Replace_steering_column')

    ########################## AC##################################
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_compressor','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_AC_COMPRESSOR','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_A/C_COMPRESSOR','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_A/C_compressor_ASSY','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_a/c_compressor_assy','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_aircon_compressor','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_A/C_compressor_assy','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACED_A/C_COMPRESSOR_ASSY','Replace_A/C_compressor')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACED_AC_COMPRESSOR','Replace_A/C_compressor')

    ################## Transmission#################################
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_transmission_assy','Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace__CVT_transmission_assy', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_TRANSMISSION_ASSEMBLY', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Transmission_Assembly', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Transmission_ASSY', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_THE_CVT_TRANSMISSION', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replaced_transmission_assembly','Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replaced_Transmission_ASM', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_CVT_transmission', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace__transmission_assy', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_TRANS_ASSY', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_transmission_assembly', 'Replace_transmission')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_transmission_', 'Replace_transmission')

    ############################ Alternator#######################
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Alternator_Assembly', 'Replace_alternator')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Alternator_assembly', 'Replace_alternator')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('alternator_assembly', 'Replace_alternator')


    ###################### Radio ##################################
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Please_Replace_failure_radio','Replace_radio')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_radio_assy','Replace_radio')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('radio_assembly','Replace_radio')
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('REPLACE_RADIO_UNIT','Replace_radio')
    ###############################################################
    data["TAC_Closing_Summing_up"] = data["TAC_Closing_Summing_up"].str.replace('Replace_Fuse_box_assembly','Replace_fuse_box_assembly')

    ################################################################
    data["TAC_Closing_Summing_up"] = (data['TAC_Closing_Summing_up']).astype(str)


    #data_filtered['class'] = (pd.factorize(data_filtered['TAC_Closing_Summing_up'])[0] + 1).astype(str)
    data['TAC_Closing_Summing_up'] = (data['TAC_Closing_Summing_up']).astype(str)

    data['TAC_Closing_Summing_up'] = data['TAC_Closing_Summing_up'].astype(str)
    print("There are: ",data['TAC_Closing_Summing_up'].nunique(), " unique TAC_Closing_Summing_up")

    print("There are: ",data['TAC_Closing_Summing_up'].value_counts(), " value_counts in TAC_Closing_Summing_up")
    
    uniq_TAC_Closing_Summing_up = data['TAC_Closing_Summing_up'].unique() #returns a list of unique values
    print(type(uniq_TAC_Closing_Summing_up))
    
    #preporssing DTC col
    data['DTC'] = data['DTC'].str.replace('_',' ')
    data['DTC'] = data['DTC'].str.replace('NO','NO-DTC')
    data['DTC'] = data['DTC'].fillna('NO-DTC', inplace=False)
    
    total_duplicate_titles = sum(data["Case No."].duplicated())
    print(f"There are {total_duplicate_titles} duplicate case no.")
    
    data = data[~data["Case No."].duplicated()]
    print(f"There are {len(data)} rows in the deduplicated dataset.")

    # There are some terms with occurrence as low as 1.
    print(sum(data["TAC_Closing_Summing_up"].value_counts() == 3))

    # How many unique terms?
    print(data["TAC_Closing_Summing_up"].nunique())
    
    # Filtering the rare terms.
    print("Shape before", data.shape)
    data_filtered = data.groupby("TAC_Closing_Summing_up").filter(lambda x: len(x) >= 3 )
    print("Shape after", data_filtered.shape)
    
    data_filtered["Fault Symptom"] = (data_filtered["Fault Symptom"] +", "+data_filtered["System 1"] +", "+ data_filtered["System Type 2"] +", "+ data_filtered["System Type 3"] +", "+ data_filtered["Affected"] +", "+ data_filtered["FaultCause"] +", "+ data_filtered["FaultPhenoment"] +", "+ data_filtered["DTC"]).astype(str)
    
    data_filtered = data_filtered.drop(['Case No.', 'Case Status', 'Case No.', 'Subject', 'Mileage', 'Materiel', 'System 1', 'System Type 2', 'System Type 3',
                          'Affected', 'FaultCause', 'FaultPhenoment', 'DTC'], axis = 1)
    
    
    
    #Use stratified splits because of class imbalance
    #The dataset has a class imbalance problem. So, to have a fair evaluation result,
    #we need to ensure the datasets are sampled with stratification.

    test_split = 0.2

    # Initial train and test split.
    train_df, test_df = train_test_split(
        data_filtered,
        test_size=test_split,
        stratify= data_filtered["TAC_Closing_Summing_up"].values,
    )

    # Splitting the test set further into validation
    # and new test sets.
    val_df = test_df.sample(frac=0.2)
    test_df.drop(val_df.index, inplace=True)

    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")
    print(f"Number of rows in test set: {len(test_df)}")
    
    train_df.to_csv(os.path.join(f"{base_dir}/train/train_df.csv"))
    test_df.to_csv(os.path.join(f"{base_dir}/test/test_df.csv"))
    val_df.to_csv(os.path.join(f"{base_dir}/test/val_df.csv")) 
