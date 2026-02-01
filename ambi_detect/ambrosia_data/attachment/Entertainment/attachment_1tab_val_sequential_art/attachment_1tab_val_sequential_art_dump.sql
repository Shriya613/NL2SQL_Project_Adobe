BEGIN TRANSACTION;
CREATE TABLE Authors(
    AuthorID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Authors" VALUES(1,'John Doe');
INSERT INTO "Authors" VALUES(2,'Jane Smith');
INSERT INTO "Authors" VALUES(3,'Alice Johnson');
INSERT INTO "Authors" VALUES(4,'Bob Brown');
INSERT INTO "Authors" VALUES(5,'Charlie White');
CREATE TABLE Comics(
    ComicID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    SequentialArt VARCHAR(255) ,
    ReleaseDate DATE,
    PublisherId INTEGER,
    AuthorId INTEGER,
    FOREIGN KEY (AuthorId) REFERENCES Authors(AuthorID),
    FOREIGN KEY (PublisherId) REFERENCES Publisher(PublisherID)
);
INSERT INTO "Comics" VALUES(1,'The Walking Dead','Comic Book','2019-08-01',1,1);
INSERT INTO "Comics" VALUES(2,'Saga','Comic Book','2016-03-17',2,2);
INSERT INTO "Comics" VALUES(3,'Watchmen','Graphic Novel','2019-02-28',1,3);
INSERT INTO "Comics" VALUES(4,'Y: The Last Man','Graphic Novel','2019-06-20',3,4);
INSERT INTO "Comics" VALUES(5,'V for Vendetta','Graphic Novel','2019-05-01',4,5);
INSERT INTO "Comics" VALUES(6,'Hellboy','Comic Book','2019-07-05',2,6);
INSERT INTO "Comics" VALUES(7,'Superman','Comic Book','2019-09-02',3,7);
CREATE TABLE Movie(
    MovieID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    DirectorName TEXT,
    ReleaseDate DATE
);
INSERT INTO "Movie" VALUES(1,'Inception','Christopher Nolan','2010-07-16');
INSERT INTO "Movie" VALUES(2,'Interstellar','Christopher Nolan','2014-11-07');
INSERT INTO "Movie" VALUES(3,'The Dark Knight','Christopher Nolan','2008-07-18');
INSERT INTO "Movie" VALUES(4,'Memento','Christopher Nolan','2000-06-22');
INSERT INTO "Movie" VALUES(5,'Fight Club','David Fincher','1999-10-15');
CREATE TABLE Publisher(
    PublisherID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Publisher" VALUES(1,'BOOM Studios');
INSERT INTO "Publisher" VALUES(2,'DC Comics');
INSERT INTO "Publisher" VALUES(3,'Marvel Comics');
INSERT INTO "Publisher" VALUES(4,'Image Comics');
INSERT INTO "Publisher" VALUES(5,'Dark Horse Comics');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publisher',5);
INSERT INTO "sqlite_sequence" VALUES('Comics',7);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
COMMIT;
