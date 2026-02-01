BEGIN TRANSACTION;
CREATE TABLE Digital_Tools(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Category TEXT,
    Access TEXT,
    Price REAL 
);
INSERT INTO "Digital_Tools" VALUES(1,'Adobe Photoshop','Graphic Design','Subscription',29.99);
INSERT INTO "Digital_Tools" VALUES(2,'Canva','Web Design','Free',0.0);
INSERT INTO "Digital_Tools" VALUES(3,'Illustrator','Graphic Design','Perpetual License',648.75);
INSERT INTO "Digital_Tools" VALUES(4,'Sketch','Web Design','Subscription',9.99);
INSERT INTO "Digital_Tools" VALUES(5,'InDesign','Graphic Design','Perpetual License',849.99);
CREATE TABLE Graphics_Designers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT ,
    Experience TEXT,
    Digital_Tool_ID INTEGER,
    FOREIGN KEY (Digital_Tool_ID) REFERENCES Digital_Tools(id)
);
INSERT INTO "Graphics_Designers" VALUES(1,'John Doe',25,'1 Year of experience',1);
INSERT INTO "Graphics_Designers" VALUES(2,'Jane Smith',30,'3 Years of experience',2);
INSERT INTO "Graphics_Designers" VALUES(3,'Mike Johnson',28,'2 Years of experience',3);
INSERT INTO "Graphics_Designers" VALUES(4,'Emily Williams',22,'Fresh Graduate',4);
INSERT INTO "Graphics_Designers" VALUES(5,'Robert Brown',35,'5 Years of experience',5);
CREATE TABLE Movie_Directors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT ,
    YearsOfExperience INT 
);
INSERT INTO "Movie_Directors" VALUES(1,'James Cameron',48,'35 years');
INSERT INTO "Movie_Directors" VALUES(2,'Christopher Nolan',51,'22 years');
INSERT INTO "Movie_Directors" VALUES(3,'Steven Spielberg',73,'49 years');
INSERT INTO "Movie_Directors" VALUES(4,'Quentin Tarantino',57,'29 years');
INSERT INTO "Movie_Directors" VALUES(5,'J. J. Abrams',54,'19 years');
CREATE TABLE Movie_Producers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Company TEXT,
    Budget REAL 
);
INSERT INTO "Movie_Producers" VALUES(1,'Marvel Studios','Marvel Cinematic Universe',1000000000.0);
INSERT INTO "Movie_Producers" VALUES(2,'Disney','Walt Disney Pictures',200000000.0);
INSERT INTO "Movie_Producers" VALUES(3,'Warner Bros.','DC Films',150000000.0);
INSERT INTO "Movie_Producers" VALUES(4,'Universal Pictures','Blumhouse Productions',80000000.0);
INSERT INTO "Movie_Producers" VALUES(5,'Paramount Pictures','DreamWorks Animation',120000000.0);
CREATE TABLE Movies(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Producer_ID INTEGER,
    Director_ID INTEGER,
    ReleaseDate DATE,
    BoxOfficeCollection REAL ,
    FOREIGN KEY (Producer_ID) REFERENCES Movie_Producers(id),
    FOREIGN KEY (Director_ID) REFERENCES Movie_Directors(id)
);
INSERT INTO "Movies" VALUES(1,'Avengers: Endgame',1,2,'2019-04-26',15000000000.0);
INSERT INTO "Movies" VALUES(2,'The Lion King',3,1,'2019-07-19',1657000000.0);
INSERT INTO "Movies" VALUES(3,'Captain Marvel',1,1,'2019-03-08',1022000000.0);
INSERT INTO "Movies" VALUES(4,'Toy Story 4',4,1,'2019-06-21',1014000000.0);
INSERT INTO "Movies" VALUES(5,'Star Wars: The Rise of Skywalker',2,1,'2019-12-20',1075000000.0);
CREATE TABLE Web_Designers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INT ,
    Experience TEXT,
    Digital_Tool_ID INTEGER,
    FOREIGN KEY (Digital_Tool_ID) REFERENCES Digital_Tools(id)
);
INSERT INTO "Web_Designers" VALUES(1,'Tom Miller',26,'2 Years of experience',2);
INSERT INTO "Web_Designers" VALUES(2,'Nancy Davis',32,'4 Years of experience',4);
INSERT INTO "Web_Designers" VALUES(3,'Chris Taylor',30,'1 Year of experience',3);
INSERT INTO "Web_Designers" VALUES(4,'Lisa Jackson',28,'Fresh Graduate',5);
INSERT INTO "Web_Designers" VALUES(5,'David White',29,'3 Years of experience',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Digital_Tools',5);
INSERT INTO "sqlite_sequence" VALUES('Graphics_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Web_Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Movie_Producers',5);
INSERT INTO "sqlite_sequence" VALUES('Movie_Directors',5);
INSERT INTO "sqlite_sequence" VALUES('Movies',5);
COMMIT;
