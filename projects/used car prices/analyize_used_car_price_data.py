'''
ideas:

classify cheap vs expensive cars into 1? 2? 3? classes? compare intra class variance. minimze variance amoung these
    - better way to analyze variance? trade off for lower variance vs class? loss function?

year vs average car price
 - kernel ridge regression on gaussian kernel vs guassian process for regression

decission trees for classification of cars?...

group average used car price per year, and make marcov chain with transision matrix,
each transition going between nodex (latent variable), with x, output, being the price (the actual year is irrelevent, since we are only concern with one year progression)
  z1 -> z2 -> z3
   |     |     |
   $     $     $
 - derive the transition matrix for this marcov chain


'''