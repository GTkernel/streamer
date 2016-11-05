import ConfigParser, os


config = ConfigParser.ConfigParser()
config.readfp(open('streamer.cfg'))

STREAMER_SERVER_HOST = config.get('server', 'host')
STREAMER_SERVER_PORT = config.get('server', 'port')
