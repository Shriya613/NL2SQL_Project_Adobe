BEGIN TRANSACTION;
CREATE TABLE Article_Publications(
    ArticleID INTEGER,
    PublicationID INTEGER,
    PublishedOn DATETIME,
    FOREIGN KEY (ArticleID) REFERENCES Articles(ArticleID),
    FOREIGN KEY (PublicationID) REFERENCES Publications(PublicationID)
);
CREATE TABLE Articles(
    ArticleID INTEGER PRIMARY KEY AUTOINCREMENT,
    Title TEXT,
    Content TEXT,
    PublishDate DATETIME,
    ReporterID INTEGER,
    FOREIGN KEY (ReporterID) REFERENCES Fashion_Reporters(ReporterID)
);
INSERT INTO "Articles" VALUES(1,'The Rise of Sustainable Fashion','...','2021-08-01 12:34:56',1);
INSERT INTO "Articles" VALUES(2,'Inside the World of High Fashion','...','2021-09-01 12:34:56',2);
INSERT INTO "Articles" VALUES(3,'Street Style: The New Frontier of Fashion','...','2021-10-01 12:34:56',3);
INSERT INTO "Articles" VALUES(4,'The Evolution of Haute Couture','...','2021-11-01 12:34:56',4);
INSERT INTO "Articles" VALUES(5,'A Look Back at Iconic Fashion Moments','...','2021-12-01 12:34:56',5);
CREATE TABLE Designers(
    DesignerID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT
);
INSERT INTO "Designers" VALUES(1,'Christian_Dior');
INSERT INTO "Designers" VALUES(2,'Gucci');
INSERT INTO "Designers" VALUES(3,'Prada');
INSERT INTO "Designers" VALUES(4,'Chanel');
INSERT INTO "Designers" VALUES(5,'Louis_Vuitton');
CREATE TABLE Fashion_Reporters(
    ReporterID INTEGER PRIMARY KEY AUTOINCREMENT,
    FullName TEXT);
INSERT INTO "Fashion_Reporters" VALUES(1,'John Smith');
INSERT INTO "Fashion_Reporters" VALUES(2,'Jane Doe');
INSERT INTO "Fashion_Reporters" VALUES(3,'Michael Brown');
INSERT INTO "Fashion_Reporters" VALUES(4,'Emily White');
INSERT INTO "Fashion_Reporters" VALUES(5,'David Gray');
CREATE TABLE Fashion_Reporters_Designers(
    ReporterID INTEGER,
    DesignerID INTEGER,
    FOREIGN KEY (ReporterID) REFERENCES Fashion_Reporters(ReporterID),
    FOREIGN KEY (DesignerID) REFERENCES Designers(DesignerID),
    PRIMARY KEY (ReporterID, DesignerID)
);
INSERT INTO "Fashion_Reporters_Designers" VALUES(1,1);
INSERT INTO "Fashion_Reporters_Designers" VALUES(2,1);
INSERT INTO "Fashion_Reporters_Designers" VALUES(3,1);
INSERT INTO "Fashion_Reporters_Designers" VALUES(4,1);
INSERT INTO "Fashion_Reporters_Designers" VALUES(5,1);
INSERT INTO "Fashion_Reporters_Designers" VALUES(5,4);
CREATE TABLE Publications(
    PublicationID INTEGER PRIMARY KEY AUTOINCREMENT,
    PublisherName TEXT,
    WebsiteURL TEXT
);
INSERT INTO "Publications" VALUES(1,'Vogue Magazine','www.vogue.com');
INSERT INTO "Publications" VALUES(2,'Harper’s Bazaar','www.harpersbazaar.com');
INSERT INTO "Publications" VALUES(3,'Elle Magazine','www.elle.com');
INSERT INTO "Publications" VALUES(4,'GQ Magazine','www.gq.com');
INSERT INTO "Publications" VALUES(5,'Esquire Magazine','www.esquire.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Designers',5);
INSERT INTO "sqlite_sequence" VALUES('Fashion_Reporters',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Publications',5);
COMMIT;
