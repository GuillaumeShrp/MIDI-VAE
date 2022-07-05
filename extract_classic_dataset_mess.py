import requests, bs4

for i in range(1,4):
    url = 'https://midkar.com/classical/classical_0' + str(i) + '.html'
    response = requests.get(url)


    parsed = bs4.BeautifulSoup(response.text, 'html.parser')

    musictab = parsed.findAll('table')[3]
    musiclist = musictab.findAll('tr')[1:] #avoid table first line
    print(len(musiclist))
    for music in musiclist:
        musictitle = music.findAll('td')[0].text  
        for x in music.findAll('td')[2].findAll('b')[-1]:
            print(x.text)
        #print(musicstyle = music.findAll('td')[2].findAll('b').text)
        
        
        musicstyle = music.findAll('td')[2].findAll('b')[-1].text
        print("title", musictitle)
        print("style", musicstyle)
        if 'Classical' in musicstyle:
            link = music.findAll('td')[0].find('a')['href']
            link_url = requests.compat.urljoin(url, link)
            if link.endswith('.mid'):
                # download midi file and write it to a local file
                filename = 'data/original/classical/' + link
                with open(filename, 'wb') as midifile:
                    midifile.write(requests.get(link_url).content)
        print("end of music")
                    