from RStask import LanduseFunction
model=LanduseFunction('cuda:0')
model.inference('./1367.png','road','./output.png')