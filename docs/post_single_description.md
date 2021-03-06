# Predict Single Beer Type

<hr>

## Purpose

Use [/beer/type](/beer/type) to query for only a single beer type.

<hr>

## Expected Input String

```
/beer/type?brewery_name={brewery_name}&review_aroma={review_aroma}&review_appearance={review_appearance}&review_palate={review_palate}&review_taste={review_taste}
```

<hr>

## Input Types:

Param | Type
---|---
`brewery_name` | `str`
`review_aroma` | `float`
`review_appearance` | `float`
`review_palate` | `float`
`review_taste` | `float`

<hr>

## Validations

Param | Validation
---|---
`brewery_name` | Must be valid brewery name.
`review_aroma` <br> `review_appearance` <br> `review_palate` <br> `review_taste` | Must all be `float` and between `0` and `5`

<hr>

## Example Input

[/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1](/beer/type?brewery_name=Epic%20Ales&review_aroma=1&review_appearance=1&review_palate=1&review_taste=1)

<hr>

## Example Output

```
American IPA
```

<hr>
