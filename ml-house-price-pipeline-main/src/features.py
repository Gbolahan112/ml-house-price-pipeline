# src/features.py

def add_features(df):
    df["Rooms_per_Household"] = df["AveRooms"] / df["AveOccup"]
    df["Bedrooms_per_Household"] = df["AveBedrms"] / df["AveOccup"]
    df["Income_per_Person"] = df["MedInc"] / df["Population"]
    
    return df