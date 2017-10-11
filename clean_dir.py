for filename in findFiles('/home/user01/dev/data/gutenberg/sub/*.txt'):
  file, file_len = helpers.read_file(filename)
  newname = '/home/user01/dev/data/gutenberg/clean/' + filename[35:]
  with open(newname, 'w') as ofile:
    ofile.write(file)
