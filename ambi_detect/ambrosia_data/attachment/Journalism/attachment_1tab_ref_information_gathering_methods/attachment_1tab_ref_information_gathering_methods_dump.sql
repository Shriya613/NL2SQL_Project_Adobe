BEGIN TRANSACTION;
CREATE TABLE Article_Source(
    article_source_id INTEGER PRIMARY KEY,
    news_article_id INTEGER,
    source_id INTEGER,
    FOREIGN KEY (news_article_id) REFERENCES News_Article(article_id),
    FOREIGN KEY (source_id) REFERENCES Source(source_id)
);
INSERT INTO "Article_Source" VALUES(1,1,1);
INSERT INTO "Article_Source" VALUES(2,2,2);
INSERT INTO "Article_Source" VALUES(3,3,3);
INSERT INTO "Article_Source" VALUES(4,4,4);
CREATE TABLE Information_Gathering_Method(
    id INTEGER PRIMARY KEY,
    method TEXT
);
INSERT INTO "Information_Gathering_Method" VALUES(1,'Press Conference');
INSERT INTO "Information_Gathering_Method" VALUES(2,'Interview');
CREATE TABLE News_Article(
    article_id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    date DATE,
    author_id INTEGER,
    information_gathering_method_id INTEGER,
    participant_type_id INTEGER,
    FOREIGN KEY (author_id) REFERENCES Author(author_id),
    FOREIGN KEY (information_gathering_method_id) REFERENCES Information_Gathering_Method(id),
    FOREIGN KEY (participant_type_id) REFERENCES Participant_Type(id)
);
INSERT INTO "News_Article" VALUES(1,'First Press Conference','Details about the press conference...','2023-01-01',1,1,1);
INSERT INTO "News_Article" VALUES(2,'Second Press Conference','Details about the press conference...','2023-01-02',2,1,2);
INSERT INTO "News_Article" VALUES(3,'Interview with Politician','Details about the interview...','2023-01-03',3,2,1);
INSERT INTO "News_Article" VALUES(4,'Interview with Activist','Details about the activist interview...','2023-01-04',4,2,2);
CREATE TABLE Participant_Type(
    id INTEGER PRIMARY KEY,
    type TEXT
);
INSERT INTO "Participant_Type" VALUES(1,'Politician');
INSERT INTO "Participant_Type" VALUES(2,'Business Leader');
INSERT INTO "Participant_Type" VALUES(3,'Activist');
CREATE TABLE Quote(
    quote_id INTEGER PRIMARY KEY,
    text TEXT,
    start_time TIME,
    end_time TIME,
    news_article_id INTEGER,
    speaker_id INTEGER,
    FOREIGN KEY (news_article_id) REFERENCES News_Article(article_id),
    FOREIGN KEY (speaker_id) REFERENCES Speaker(speaker_id)
);
INSERT INTO "Quote" VALUES(1,'Quote from politician at first press conference.','00:00:05','00:00:15',1,1);
INSERT INTO "Quote" VALUES(2,'Quote from business leader at second press conference.','00:00:06','00:00:17',2,2);
INSERT INTO "Quote" VALUES(3,'Quote from politician during the interview.','00:00:08','00:00:19',3,3);
INSERT INTO "Quote" VALUES(4,'Quote from activist during the interview.','00:00:09','00:00:21',4,4);
CREATE TABLE Source(
    source_id INTEGER PRIMARY KEY,
    name TEXT,
    url TEXT UNIQUE
);
INSERT INTO "Source" VALUES(1,'Official Government Website','https://www.govsite.com/');
INSERT INTO "Source" VALUES(2,'Company Blog','https://blog.company.com/');
INSERT INTO "Source" VALUES(3,'News Channel','https://newschannel.tv/');
INSERT INTO "Source" VALUES(4,'Alternative Media Outlet','https://alternativemedia.org/');
CREATE TABLE Speaker(
    speaker_id INTEGER PRIMARY KEY,
    name TEXT,
    role TEXT
);
INSERT INTO "Speaker" VALUES(1,'John Doe','Politician');
INSERT INTO "Speaker" VALUES(2,'Jane Smith','Business Leader');
INSERT INTO "Speaker" VALUES(3,'Bob Johnson','Politician');
INSERT INTO "Speaker" VALUES(4,'Emma Brown','Activist');
COMMIT;
