BEGIN TRANSACTION;
CREATE TABLE article_details (
    detailID INTEGER PRIMARY KEY AUTOINCREMENT,
    details TEXT,
    articleID INT REFERENCES articles(articleID)
);
INSERT INTO "article_details" VALUES(1,'Key takeaways from the interview.',1);
INSERT INTO "article_details" VALUES(2,'Impacts on food security around the globe.',2);
INSERT INTO "article_details" VALUES(3,'New developments in photovoltaic cell efficiency.',3);
INSERT INTO "article_details" VALUES(4,'Colonization efforts for future human settlements.',4);
INSERT INTO "article_details" VALUES(5,'Government incentives for EV adoption.',5);
CREATE TABLE articles (
    articleID INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publishDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    author INT REFERENCES authors(authorID)
);
INSERT INTO "articles" VALUES(1,'The world''s largest robots','This article discusses the latest advancements in robotics technology...','2024-03-13 20:57:42',NULL);
INSERT INTO "articles" VALUES(2,'Climate change and its impact on agriculture','A comprehensive analysis of how climate change is affecting global agricultural practices...','2024-03-13 20:57:42',NULL);
INSERT INTO "articles" VALUES(3,'Revolutionary breakthroughs in renewable energy','An exploration of recent innovations in solar, wind, and other green energy sources...','2024-03-13 20:57:42',NULL);
INSERT INTO "articles" VALUES(4,'Advancements in space exploration','Examining the progress made in sending humans to Mars and beyond...','2024-03-13 20:57:42',NULL);
INSERT INTO "articles" VALUES(5,'The rise of electric vehicles','A look at the growing popularity of electric cars and their implications for the automotive industry...','2024-03-13 20:57:42',NULL);
CREATE TABLE authors (
    authorID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE
);
INSERT INTO "authors" VALUES(1,'John Doe');
INSERT INTO "authors" VALUES(2,'Jane Smith');
INSERT INTO "authors" VALUES(3,'Alice Johnson');
INSERT INTO "authors" VALUES(4,'Bob Brown');
INSERT INTO "authors" VALUES(5,'Charlie Green');
CREATE TABLE photos (
    photoID INTEGER PRIMARY KEY AUTOINCREMENT,
    "photo" TEXT,
    description TEXT,
    articleID INT REFERENCES articles(articleID),
    FOREIGN KEY (photoID) REFERENCES photos(photoID) ON DELETE CASCADE
);
INSERT INTO "photos" VALUES(1,'images/robot_1.jpg','Closeup image of a giant robot.',1);
INSERT INTO "photos" VALUES(2,'images/agriculture_field.jpg','Lush fields of crops being affected by drought.',2);
INSERT INTO "photos" VALUES(3,'images/wind_farm.jpg','Wind turbines generating clean energy against a sunset backdrop.',3);
INSERT INTO "photos" VALUES(4,'images/mars_lander.jpg','Mission control watching as a rover lands on Mars.',4);
INSERT INTO "photos" VALUES(5,'images/tesla_model_s.jpg','Electric car charging station along a highway.',5);
CREATE TABLE videos (
    videoID INTEGER PRIMARY KEY AUTOINCREMENT,
    "video" TEXT,
    description TEXT,
    articleID INT REFERENCES articles(articleID),
    FOREIGN KEY (videoID) REFERENCES videos(videoID) ON DELETE CASCADE
);
INSERT INTO "videos" VALUES(1,'https://www.youtube.com/watch?v=abcdefg123','Interview with a leading robotics expert.',1);
INSERT INTO "videos" VALUES(2,'https://www.youtube.com/watch?v=hijklmnopqr','Documentary on the effects of climate change on farming communities.',2);
INSERT INTO "videos" VALUES(3,'https://www.youtube.com/watch?v=stuvwxyz123','Innovative solar panel designs for residential use.',3);
INSERT INTO "videos" VALUES(4,'https://www.youtube.com/watch?v=9876543210','SpaceX launching another successful mission to Mars.',4);
INSERT INTO "videos" VALUES(5,'https://www.youtube.com/watch?v=jkmnlpqrtu','Tesla Model S acceleration test on a racetrack.',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('authors',5);
INSERT INTO "sqlite_sequence" VALUES('articles',5);
INSERT INTO "sqlite_sequence" VALUES('photos',5);
INSERT INTO "sqlite_sequence" VALUES('videos',5);
INSERT INTO "sqlite_sequence" VALUES('article_details',5);
COMMIT;
