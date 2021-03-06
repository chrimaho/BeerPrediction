# Predict Single Beer Type

_Purpose_: <br>
Use [/beer/type](/beer/type) to query for only a single beer type.

_Expected Input String_: <br>
```
/beers/type?brewery_name=Epic%20Ales&brewery_name=Epic%20Ales&review_aroma=1&review_aroma=1&review_appearance=1&review_appearance=1&review_palate=1&review_palate=1&review_taste=1&review_taste=1
```

_Input Types_: <br>
As defined below. Specifically:
1. brewery_name: list of str
1. review_aroma: list of float
1. review_appearance: list of float
1. review_palate: list of float
1. review_taste: list of float

_Validations_: <br>
1. `brewery_name`: Must be valid brewery names.
1. `review_aroma`, `review_appearance`, `review_palate`, `review_taste`: Must all be `float` values, between `0` and `5`.
1. The length of all the parameters must be the same.

_Example Input_: <br>
[/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1](/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1)

_Example Output_: <br>
```
[
  "(512) Brewing Company",
  "(512) Brewing Company"
]
```