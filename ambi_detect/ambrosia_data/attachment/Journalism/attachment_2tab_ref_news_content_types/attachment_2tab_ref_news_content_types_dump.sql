BEGIN TRANSACTION;
CREATE TABLE Authors(
    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Authors" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Authors" VALUES(3,'Alice Johnson','alicejohnson@example.com');
INSERT INTO "Authors" VALUES(4,'Bob Brown','bobbrown@example.com');
INSERT INTO "Authors" VALUES(5,'Carol White','carolwhite@example.com');
CREATE TABLE Feature_Articles(
    feature_article_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    volume_id INTEGER,
    FOREIGN KEY (volume_id) REFERENCES Volumes(volume_id));
INSERT INTO "Feature_Articles" VALUES(1,'The History of Newspapers','An exploration of newspaper history dating back centuries...',1);
INSERT INTO "Feature_Articles" VALUES(2,'Investigative Journalism Techniques','Discovering the best practices in investigative journalism...',2);
INSERT INTO "Feature_Articles" VALUES(3,'Censorship Around The World','Looking at censorship laws around the globe...',3);
INSERT INTO "Feature_Articles" VALUES(4,'Journalism Ethics and Standards','Understanding the ethical standards in modern journalism...',4);
INSERT INTO "Feature_Articles" VALUES(5,'Future Of News Media','Predicting trends and changes within the news media industry...',5);
CREATE TABLE Opinion_Pieces(
    opinion_piece_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    volume_id INTEGER,
    FOREIGN KEY (volume_id) REFERENCES Volumes(volume_id));
INSERT INTO "Opinion_Pieces" VALUES(1,'The Importance of Free Press','A discussion about freedom of speech and press...',1);
INSERT INTO "Opinion_Pieces" VALUES(2,'Technology and Society','How technology has shaped our society today.',2);
INSERT INTO "Opinion_Pieces" VALUES(3,'Economic Growth and Poverty','Exploring how economic growth impacts poverty levels...',3);
INSERT INTO "Opinion_Pieces" VALUES(4,'Climate Change and Its Effects','Discussing the effects of climate change on our planet...',4);
INSERT INTO "Opinion_Pieces" VALUES(5,'Healthcare Accessibility','Analyzing healthcare accessibility across various demographics...',5);
CREATE TABLE Publications(
    publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT);
INSERT INTO "Publications" VALUES(1,'New York Times','1 New York Times Square, New York City, NY 10007');
INSERT INTO "Publications" VALUES(2,'Washington Post','1301 K Street NW, Washington DC 20071');
INSERT INTO "Publications" VALUES(3,'Los Angeles Times','202 W 1st St #600, Los Angeles CA 90012');
INSERT INTO "Publications" VALUES(4,'Chicago Tribune','435 N Michigan Ave, Chicago IL 60611');
INSERT INTO "Publications" VALUES(5,'Miami Herald','3511 NW 91 Avenue, Doral FL 33172');
CREATE TABLE Volumes(
    volume_id INTEGER PRIMARY KEY AUTOINCREMENT,
    number TEXT ,
    publication_date DATE);
INSERT INTO "Volumes" VALUES(1,'01','2023-01-01');
INSERT INTO "Volumes" VALUES(2,'02','2023-02-01');
INSERT INTO "Volumes" VALUES(3,'03','2023-03-01');
INSERT INTO "Volumes" VALUES(4,'04','2023-04-01');
INSERT INTO "Volumes" VALUES(5,'05','2023-05-01');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Volumes',5);
INSERT INTO "sqlite_sequence" VALUES('Opinion_Pieces',5);
INSERT INTO "sqlite_sequence" VALUES('Feature_Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
COMMIT;
