"""
Pydantic schemas for the Ames ML API.
Defines the structure of requests and responses.
"""


from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Schema for housing price prediction request."""

    OverallQual: int = Field(
        ..., ge=1, le=10, description="Overall material and finish quality (1-10)"
    )
    GrLivArea: float = Field(..., ge=0, description="Above grade (ground) living area square feet")
    BsmtQual: str = Field(..., description="Height of the basement (Ex, Gd, TA, Fa, Po, NA)")
    Neighborhood: str = Field(..., description="Physical locations within Ames city limits")
    KitchenQual: str = Field(..., description="Kitchen quality (Ex, Gd, TA, Fa, Po)")
    BsmtFinSF1: float = Field(..., ge=0, description="Type 1 finished square feet")
    TotalBsmtSF: float = Field(..., ge=0, description="Total square feet of basement area")
    FirstFlrSF: float = Field(..., alias="1stFlrSF", ge=0, description="First Floor square feet")
    GarageArea: float = Field(..., ge=0, description="Size of garage in square feet")
    FullBath: int = Field(..., ge=0, description="Full bathrooms above grade")
    MasVnrArea: float = Field(..., ge=0, description="Masonry veneer area in square feet")
    ExterQual: str = Field(..., description="Exterior material quality (Ex, Gd, TA, Fa, Po)")
    YearRemodAdd: int = Field(..., alias="YearRemod/Add", ge=1800, description="Remodel date")
    MSSubClass: int = Field(..., description="The type of dwelling involved in the sale")
    YearBuilt: int = Field(..., ge=1800, description="Original construction date")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500.0,
                "BsmtQual": "Gd",
                "Neighborhood": "CollgCr",
                "KitchenQual": "Gd",
                "BsmtFinSF1": 500.0,
                "TotalBsmtSF": 1000.0,
                "1stFlrSF": 1000.0,
                "GarageArea": 500.0,
                "FullBath": 2,
                "MasVnrArea": 100.0,
                "ExterQual": "Gd",
                "YearRemod/Add": 2005,
                "MSSubClass": 60,
                "YearBuilt": 2000,
            }
        },
    )


class PredictionResponse(BaseModel):
    """Schema for housing price prediction response."""

    SalePrice: float = Field(..., description="Predicted sale price of the house")
