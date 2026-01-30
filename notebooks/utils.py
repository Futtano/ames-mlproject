# Check for wrong values inside the dataset
def validate_numerical_features(df, allowed_values):
    """Validate all rows against allowed numerical bounds for numerical features"""
    violations = {}

    for col, check_func in allowed_values.items():
        if col in df.columns:
            invalid_mask = ~(check_func(df[col]) | df[col].isna()) # Do not flag NaN as invalid
            
            if invalid_mask.any():
                if invalid_mask.any():
                    violations[col] = {
                        'count': invalid_mask.sum(),
                        'invalid_values': df.loc[invalid_mask, col].unique().tolist()
                    }
        # else:
            # raise KeyError(f'specified column {col} of "allowed_values" is not present in df.columns')
        
    return violations

def validate_categorical_features(df, allowed_values):
    """Validate all rows against allowed categories for categorical features"""
    violations = {}
    
    for col, valid_vals in allowed_values.items():
        if col in df.columns:
            invalid_mask = ~(df[col].isin(valid_vals) | df[col].isna())
            if invalid_mask.any():
                violations[col] = {
                    'count': invalid_mask.sum(),
                    'invalid_values': df.loc[invalid_mask, col].unique().tolist()
                }
        # else:
            # raise KeyError(f'specified column {col} of "allowed_values" is not present in df.columns')

    
    return violations

name_map = {
    'MS SubClass':  'MSSubClass',
    'MS Zoning': 'MSZoning',
    'Lot Shape' : 'LotShape',
    'Land Contour' : 'LandContour',
    'Lot Config' : 'LotConfig',
    'Lot Frontage': 'LotFrontage',
    'Lot Area': 'LotArea',
    'Land Slope' : 'LandSlope',
    'Condition 1' : 'Condition1',
    'Condition 2' : 'Condition2',
    'BsmtFin SF 1': 'BsmtFinSF1',
    'BsmtFin SF 2': 'BsmtFinSF2',
    'Bsmt Unf SF': 'BsmtUnfSF',
    'Total Bsmt SF': 'TotalBsmtSF',
    '1st Flr SF': '1stFlrSF',
    '2nd Flr SF': '2ndFlrSF',
    'Low Qual Fin SF': 'LowQualFinSF',
    'Gr Liv Area': 'GrLivArea',
    'Bsmt Full Bath': 'BsmtFullBath',
    'Bsmt Half Bath': 'BsmtHalfBath', 
    'Full Bath': 'FullBath', 
    'Half Bath': 'HalfBath',
    'Bedroom AbvGr': 'BedroomAbvGr',
    'Kitchen AbvGr': 'KitchenAbvGr',
    'Bldg Type' : 'BldgType',
    'House Style' : 'HouseStyle',
    'Overall Qual' : 'OverallQual',
    'Overall Cond' : 'OverallCond',
    'Roof Style' : 'RoofStyle',
    'Roof Matl' : 'RoofMatl',
    'Exterior 1st' : 'Exterior1st',
    'Exterior 2nd' : 'Exterior2nd',
    'Mas Vnr Type' : 'MasVnrType',
    'Mas Vnr Area' : 'MasVnrArea',
    'Exter Qual' : 'ExterQual',
    'Exter Cond' : 'ExterCond',
    'Bsmt Qual' : 'BsmtQual',
    'Bsmt Cond' : 'BsmtCond',
    'Bsmt Exposure' : 'BsmtExposure',
    'BsmtFin Type 1' : 'BsmtFinType1',
    'BsmtFin Type 2' : 'BsmtFinType2',
    'Heating QC' : 'HeatingQC',
    'Central Air' : 'CentralAir',
    'Kitchen Qual' : 'KitchenQual',
    'Fireplace Qu' : 'FireplaceQu',
    'Garage Type' : 'GarageType',
    'Garage Finish' : 'GarageFinish',
    'Garage Qual' : 'GarageQual',
    'Garage Cond' : 'GarageCond',
    'Garage Yr Blt': 'GarageYrBlt',
    'Garage Cars': 'GarageCars',
    'Garage Area': 'GarageArea',
    'Paved Drive' : 'PavedDrive',
    'Pool QC' : 'PoolQC',
    'Misc Feature' : 'MiscFeature',
    'Sale Type' : 'SaleType',
    'Sale Condition': 'SaleCondition',
    'TotRms AbvGrd': 'TotRmsAbvGrd',
    'Wood Deck SF': 'WoodDeckSF',
    'Open Porch SF': 'OpenPorchSF',
    'Enclosed Porch': 'EnclosedPorch',
    '3Ssn Porch': '3SsnPorch',
    'Screen Porch': 'ScreenPorch',
    'Pool Area': 'PoolArea',
    'Year Built': 'YearBuilt',
    'Year Remod/Add': 'YearRemod/Add',
    'Misc Val': 'MiscVal',
    'Mo Sold': 'MoSold',
    'Yr Sold': 'YrSold'
}

categorical_invariants = {
    'MSSubClass': [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
    'MSZoning': ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
    'Street': ['Grvl', 'Pave'],
    'Alley': ['Grvl', 'Pave', 'NA'],
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'LandContour': ['Lvl', 'Bnk', 'HLS', 'Low'],
    'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
    'LotConfig': ['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
    'LandSlope': ['Gtl', 'Mod', 'Sev'],
    'Neighborhood': ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide',
                      'ClearCr', 'CollgCr', 'Crawfor', 'Edwards',
                      'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
                      'NAmes', 'NoRidge', 'NPkVill', 'NridgHt',
                      'NWAmes', 'OldTown', 'SWISU', 'Sawyer',
                      'SawyerW', 'Somerst', 'StoneBr', 'Timber',
                      'Veenker', 'Greens', 'GrnHill', 'Landmrk'],
    'Condition1': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn',
                   'PosN', 'PosA', 'RRNe', 'RRAe'],
    'Condition2': ['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn',
                   'PosN', 'PosA', 'RRNe', 'RRAe'],
    'BldgType': ['1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'],
    'HouseStyle': ['1Story', '1.5Fin', '1.5Unf', '2Story',
                   '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'],
    'OverallQual': [i for i in range(10, 0, -1)],
    'OverallCond': [i for i in range(10, 0, -1)],
    'RoofStyle': ['Flat', 'Gable', 'Gambrel', 'Hip',
                  'Mansard', 'Shed'],
    'RoofMatl': ['ClyTile', 'CompShg', 'Membran', 'Metal',
                 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'],
    'Exterior1st': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace',
                    'CBlock', 'CemntBd', 'HdBoard', 'ImStucc',
                    'MetalSd', 'Other', 'Plywood', 'PreCast', 
                    'Stone', 'Stucco', 'VinylSd', 'WdSdng', 
                    'WdShng'],
    'Exterior2nd': ['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace',
                    'CBlock', 'CemntBd', 'HdBoard', 'ImStucc',
                    'MetalSd', 'Other', 'Plywood', 'PreCast', 
                    'Stone', 'Stucco', 'VinylSd', 'WdSdng', 
                    'WdShng'],
    'MasVnrType': ['BrkCmn', 'BrkFace', 'CBlock', 'None',
                   'Stone'],
    'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Foundation': ['BrkTil', 'CBlock', 'PConc', 'Slab',
                   'Stone', 'Wood'],
    'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
    'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
    'Heating': ['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'],
    'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'CentralAir': ['N', 'Y'],
    'Electrical': ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1',
                   'Maj2', 'Sev', 'Sal'],
    'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageType': ['2Types', 'Attchd', 'Basment', 'BuiltIn',
                   'CarPort', 'Detchd', 'NA'],
    'GarageFinish': ['Fin', 'RFn', 'Unf', 'NA'],
    'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
    'PavedDrive': ['Y', 'P', 'N'],
    'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
    'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'],
    'MiscFeature': ['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'],
    'SaleType': ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con',
                 'ConLw', 'ConLI', 'ConLD', 'Oth'],
    'SaleCondition': ['Normal', 'Abnorml', 'AdjLand',
                      'Alloca', 'Family', 'Partial']
}

numerical_invariants = {
    'LotFrontage': lambda x: x >= 0,
    'LotArea': lambda x: x >= 0,
    'YearBuilt': lambda x: x >= 1800, 
    'YearRemod/Add': lambda x: x >= 1800, 
    'GarageYrBlt': lambda x: x >= 1800,
    'MasVnrArea': lambda x: x >= 0,
    'BsmtFinSF1': lambda x: x >= 0,
    'BsmtFinSF2': lambda x: x >= 0,
    'TotalBsmtSF': lambda x: x >= 0,
    'BsmtUnfSF': lambda x: x >= 0,
    '1stFlrSF': lambda x: x >= 0,
    '2ndFlrSF': lambda x: x >= 0,
    'LowQualFinSF': lambda x: x >= 0,
    'GrLivArea': lambda x: x >= 0,
    'BsmtFullBath': lambda x: x >= 0,
    'BsmtHalfBath': lambda x: x >= 0,
    'FullBath': lambda x: x >= 0,
    'HalfBath': lambda x: x >= 0,
    'BedroomAbvGr': lambda x: x >= 0,
    'KitchenAbvGr': lambda x: x >= 0,
    'TotRmsAbvGrd': lambda x: x >= 0,
    'Fireplaces': lambda x: x >= 0,
    'GarageCars': lambda x: x >= 0,
    'GarageArea': lambda x: x >= 0,
    'WoodDeckSF': lambda x: x >= 0,
    'OpenPorchSF': lambda x: x >= 0,
    'EnclosedPorch': lambda x: x >= 0,
    '3SsnPorch': lambda x: x >= 0,
    'ScreenPorch': lambda x: x >= 0,
    'PoolArea': lambda x: x >= 0,
    'MiscVal': lambda x: x >= 0,
    'MoSold': lambda x: (x >= 1) & (x <= 12),
    'YrSold': lambda x: x >= 1800,
    'SalePrice': lambda x: x >= 0,
}

mszoning_correct_values = {
    'C (all)': 'C',
    'I (all)': 'I',
    'A (agr)': 'A',
}

bldgtype_correct_values = {
    'Twnhs': 'TwnhsI',
    'Duplex': 'Duplx',
    '2fmCon': '2FmCon'
}

exterior1st_correct_values = {
    'Wd Sdng': 'WdSdng',
    'Wd Shing': 'WdShng',
    'WdShing': 'WdShng'
}

exterior2d_correct_values = {
    'CmentBd': 'CemntBd',
    'Brk Cmn': 'BrkComm',
    'Wd Sdng': 'WdSdng',
    'Wd Shng': 'WdShng'
}

saletype_correct_values = {
    'WD ': 'WD'
}

ordered_categories = [
    'Utilities', 'LandSlope', 'OverallQual', 'OverallCond',
    'ExterQual', 'ExterCond', 'Electrical', 'LotShape',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional',
    'FireplaceQu', 'GarageYrBuilt', 'GarageFinish', 'GarageQual',
    'GarageCond', 'PoolQC', 'Fence', 'PavedDrive'
]

float_feat = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
              'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
              'GrLivArea', 'GarageArea','WoodDeckSF','OpenPorchSF', 'EnclosedPorch',
              '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']
int_feat = ['YearBuilt', 'YearRemod/Add', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
             'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
             'GarageYrBlt', 'GarageCars', 'MoSold', 'YrSold']