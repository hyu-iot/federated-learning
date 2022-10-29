# https://coding-moomin.tistory.com/24

import ftplib

session = ftplib.FTP()
session.connect('127.0.0.1', 2121)
session.login('user', '12345')

uploadfile = open('./data/sample.txt', mode='rb')

session.encoding='utf-8'
session.storbinary('STOR ' + '/result/savedSample.txt', uploadfile)

uploadfile.close()
session.quit()
print('Send File')