
def external_synth(new_file):
    """
        Sends OSC
        Sends OSC to external synthesis engine like SuperCollider or pd
    """
    print("\tPlaying %s"%new_file)
    msg = OSC.OSCMessage()
    msg.setAddress("/play")

    #mac os #FIXME
    msg.append( "apicultor"+new_file.split('.')[1]+'.wav' )

    try:
        osc_client.send(msg)
    except Exception,e:
        print(e)
    #TODO: get duration from msg (via API)
    time.sleep(duration)
#external_synth()
