BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Birthdate DATE,
    Biography TEXT
);
INSERT INTO "Authors" VALUES(1,'John','Doe','1980-01-01','An experienced journalist working for Global Times.');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','1982-02-01','Known for her investigative pieces, Jane works for News Spectrum.');
INSERT INTO "Authors" VALUES(3,'Michael','Brown','1983-03-01','With a background in digital media, Michael contributes to Journal Worldwide.');
INSERT INTO "Authors" VALUES(4,'Emily','Davis','1984-04-01','Specializing in social issues reporting, Emily writes for Globe Insight.');
INSERT INTO "Authors" VALUES(5,'William','Johnson','1985-05-01','Focused on breaking news stories, William has made a name for himself at World Watcher.');
CREATE TABLE Feature_Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    PublicationDate DATE,
    Volume TEXT
);
INSERT INTO "Feature_Articles" VALUES(1,'A Day in the Life of a Reporter','Alice Green','2023-06-01','01');
INSERT INTO "Feature_Articles" VALUES(2,'The Evolution of Photojournalism','Bob White','2023-07-01','02');
INSERT INTO "Feature_Articles" VALUES(3,'How Data Analysis is Changing Journalism','Carol Black','2023-08-01','03');
INSERT INTO "Feature_Articles" VALUES(4,'Behind the Scenes at a Breaking News Desk','David Wilson','2023-09-01','04');
INSERT INTO "Feature_Articles" VALUES(5,'Understanding Fake News','Eva Young','2023-10-01','05');
CREATE TABLE Magazines(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Country TEXT,
    FoundedYear INT
);
INSERT INTO "Magazines" VALUES(1,'Global Times','USA',2000);
INSERT INTO "Magazines" VALUES(2,'World Watcher','Canada',2005);
INSERT INTO "Magazines" VALUES(3,'News Spectrum','UK',2010);
INSERT INTO "Magazines" VALUES(4,'Journal Worldwide','Australia',2015);
INSERT INTO "Magazines" VALUES(5,'Globe Insight','Germany',2020);
CREATE TABLE Opinion_Pieces(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Author TEXT,
    PublicationDate DATE,
    Volume TEXT
);
INSERT INTO "Opinion_Pieces" VALUES(1,'The Importance of Free Press','John Doe','2023-01-01','01');
INSERT INTO "Opinion_Pieces" VALUES(2,'Censorship in Modern Society','Jane Smith','2023-02-01','02');
INSERT INTO "Opinion_Pieces" VALUES(3,'Digital Media Revolution','Michael Brown','2023-03-01','03');
INSERT INTO "Opinion_Pieces" VALUES(4,'Impact of Social Media on News Consumption','Emily Davis','2023-04-01','04');
INSERT INTO "Opinion_Pieces" VALUES(5,'Investigative Reporting Today','William Johnson','2023-05-01','05');
CREATE TABLE Publishers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT,
    EstablishedYear INT
);
INSERT INTO "Publishers" VALUES(1,'Media Group Inc.','New York, USA',1995);
INSERT INTO "Publishers" VALUES(2,'Atlantic Publishing House','Toronto, Canada',2000);
INSERT INTO "Publishers" VALUES(3,'European Publisher Ltd.','London, UK',2005);
INSERT INTO "Publishers" VALUES(4,'Down Under Publishing Co.','Sydney, Australia',2010);
INSERT INTO "Publishers" VALUES(5,'Continental Press Group','Berlin, Germany',2015);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Opinion_Pieces',5);
INSERT INTO "sqlite_sequence" VALUES('Feature_Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Magazines',5);
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Publishers',5);
COMMIT;
