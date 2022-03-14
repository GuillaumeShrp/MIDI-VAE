import requests, bs4

for i in range(1,14):
    if (i < 10):
        url = 'https://midkar.com/jazz/jazz_0' + str(i) + '.html'
    else:
        url = 'https://midkar.com/jazz/jazz_' + str(i) + '.html'
    response = requests.get(url)


    parsed = bs4.BeautifulSoup(response.text, 'html.parser')

    musictab = parsed.findAll('table')[1]
    musiclist = musictab.findAll('tr')[1:] #avoid table first line
    for music in musiclist:
        musictitle = music.findAll('td')[0].text  
        musicstyle = music.findAll('td')[2].findAll('b')[-1].text
        if 'Jazz' in musicstyle:
            link = music.findAll('td')[0].find('a')['href']
            link_url = requests.compat.urljoin(url, link)
            if link.endswith('.mid'):
                # download midi file and write it to a local file
                filename = 'data/original/jazz/' + link
                with open(filename, 'wb') as midifile:
                    midifile.write(requests.get(link_url).content)
                    