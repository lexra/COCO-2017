import sys, getopt
import numpy as np
from PIL import Image
import os

"""
def main(argv):
   inputfile = ''
   outputfile = ''
   opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   for opt, arg in opts:
      if opt == '-h':
         print ('change24.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print ('Input file is ', inputfile)
   print ('Output file is ', outputfile)
   #img = Image.open(inputfile).convert('RGB')
   #img.save(outputfile)
   """

#if __name__ == "__main__":
#   main(sys.argv[1:])


import sys

IF = sys.argv[1]
OF = sys.argv[2]
#name3 = sys.argv[3]

#print("hello",  OF)
img = Image.open(IF).convert('RGB')
img.save(OF)
