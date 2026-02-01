BEGIN TRANSACTION;
CREATE TABLE Articles (
    articleID INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    publishDate DATE,
    authorID INTEGER,
    FOREIGN KEY (authorID) REFERENCES Authors (authorID)
);
INSERT INTO "Articles" VALUES(1,'The Importance of Investigative Journalism','A deep dive into how investigative journalism can uncover corruption and hold power accountable.','2023-01-01',1);
INSERT INTO "Articles" VALUES(2,'The Future of Digital Media','How digital media is changing the landscape of modern journalism.','2023-01-15',2);
INSERT INTO "Articles" VALUES(3,'Fact-Checking in a Post-Truth World','Exploring the role of fact-checkers in an era where misinformation spreads like wildfire.','2023-01-22',3);
INSERT INTO "Articles" VALUES(4,'Journalistic Ethics: A Code of Conduct','Understanding the ethical standards that journalists must adhere to.','2023-01-29',4);
INSERT INTO "Articles" VALUES(5,'Censorship and Freedom of the Press','A discussion on the challenges faced by journalists in countries with limited press freedom.','2023-02-05',5);
CREATE TABLE Authors (
    authorID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    email VARCHAR(255) UNIQUE
);
INSERT INTO "Authors" VALUES(1,'John','Doe','john.doe@example.com');
INSERT INTO "Authors" VALUES(2,'Jane','Smith','jane.smith@example.com');
INSERT INTO "Authors" VALUES(3,'Michael','Brown','michael.brown@example.com');
INSERT INTO "Authors" VALUES(4,'Emily','White','emily.white@example.com');
INSERT INTO "Authors" VALUES(5,'David','Green','david.green@example.com');
CREATE TABLE Demographics (
    demographicID INTEGER PRIMARY KEY,
    demographic_group TEXT,
    professional_field TEXT,
    magazineID INTEGER,
    FOREIGN KEY (magazineID) REFERENCES Magazines (magazineID)
);
INSERT INTO "Demographics" VALUES(1,'General Readership','Business Professional',1);
INSERT INTO "Demographics" VALUES(2,'Technology Enthusiasts','IT Specialist',2);
INSERT INTO "Demographics" VALUES(3,'Current Affairs','Political Analyst',3);
INSERT INTO "Demographics" VALUES(4,'Ethics Focused','Lawyer',4);
INSERT INTO "Demographics" VALUES(5,'International Relations','Diplomat',5);
CREATE TABLE MagazineArticles (
    magazineArticleID INTEGER PRIMARY KEY,
    articleID INTEGER,
    magazineID INTEGER,
    publicationMonth INTEGER ,
    publicationYear INTEGER ,
    FOREIGN KEY (articleID) REFERENCES Articles (articleID),
    FOREIGN KEY (magazineID) REFERENCES Magazines (magazineID)
);
INSERT INTO "MagazineArticles" VALUES(1,1,1,1,2023);
INSERT INTO "MagazineArticles" VALUES(2,2,1,2,2023);
INSERT INTO "MagazineArticles" VALUES(3,3,2,1,2023);
INSERT INTO "MagazineArticles" VALUES(4,4,2,2,2023);
INSERT INTO "MagazineArticles" VALUES(5,5,3,1,2023);
CREATE TABLE Magazines (
    magazineID INTEGER PRIMARY KEY,
    name TEXT,
    publisher TEXT,
    ISSN VARCHAR(10) UNIQUE
);
INSERT INTO "Magazines" VALUES(1,'World News Weekly','Global Publishing House','978-3-16-148410-0');
INSERT INTO "Magazines" VALUES(2,'Investigate This!','Journalism Watchdog Association','978-3-16-148411-7');
INSERT INTO "Magazines" VALUES(3,'Digital Frontiers','Tech Innovators Network','978-3-16-148412-4');
INSERT INTO "Magazines" VALUES(4,'Ethical Matters','Media Standards Council','978-3-16-148413-1');
INSERT INTO "Magazines" VALUES(5,'Press Freedoms International','Freedom Advocacy Group','978-3-16-148414-8');
CREATE TABLE Subscriptions (
    subscriptionID INTEGER PRIMARY KEY,
    customerName TEXT,
    customerEmail VARCHAR(255),
    startDate DATE,
    endDate DATE,
    magazineID INTEGER,
    FOREIGN KEY (magazineID) REFERENCES Magazines (magazineID)
);
INSERT INTO "Subscriptions" VALUES(1,'Alice Johnson','alice.johnson@example.com','2023-01-01','2023-12-31',1);
INSERT INTO "Subscriptions" VALUES(2,'Bob Smith','bob.smith@example.com','2023-01-15','2023-12-31',2);
INSERT INTO "Subscriptions" VALUES(3,'Carol Brown','carol.brown@example.com','2023-01-22','2023-12-31',3);
INSERT INTO "Subscriptions" VALUES(4,'Daniel Green','daniel.green@example.com','2023-01-29','2023-12-31',4);
INSERT INTO "Subscriptions" VALUES(5,'Emma White','emma.white@example.com','2023-02-05','2023-12-31',5);
COMMIT;
