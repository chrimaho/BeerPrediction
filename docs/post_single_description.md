# Predict Single Beer Type

_Purpose_: <br>
Use [/beer/type](/beer/type) to query for only a single beer type.

_Expected Input String_: <br>
```
/beer/type?brewery_name={brewery_name}&review_aroma={review_aroma}&review_appearance={review_appearance}&review_palate={review_palate}&review_taste={review_taste}
```

_Input Types_: <br>
As defined below. Specifically:
1. brewery_name: str
1. review_aroma: float
1. review_appearance: float
1. review_palate: float
1. review_taste: float

_Validations_: <br>
1. `brewery_name`: Must be valid brewery name.
1. `review_aroma`, `review_appearance`, `review_palate`, `review_taste`: Must all be `float` values, between `0` and `5`.

_Example Input_: <br>
[/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1](/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1)

_Example Output_: <br>
```
'(512) Brewing Company'
```