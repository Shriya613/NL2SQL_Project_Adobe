BEGIN TRANSACTION;
CREATE TABLE Authors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE);
INSERT INTO "Authors" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Authors" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Authors" VALUES(3,'Michael Brown','michael@example.com');
INSERT INTO "Authors" VALUES(4,'Emily Johnson','emily@example.com');
INSERT INTO "Authors" VALUES(5,'William Wilson','william@example.com');
CREATE TABLE Breaking_News (
    newsId INTEGER PRIMARY KEY,
    headline TEXT,
    content TEXT,
    authorId INTEGER,
    eventId INTEGER,
    FOREIGN KEY(authorId) REFERENCES Authors(id),
    FOREIGN KEY(eventId) REFERENCES Events(eventId));
INSERT INTO "Breaking_News" VALUES(1,'Earthquake Strikes Turkey and Syria','A powerful earthquake struck the border region between Turkey and Syria...',1,1);
INSERT INTO "Breaking_News" VALUES(2,'New Climate Change Report Released','The new climate change report highlights the urgent need for action...',2,3);
INSERT INTO "Breaking_News" VALUES(3,'European Political Crisis Intensifies','Tensions continue to rise as the political crisis in Europe deepens...',3,4);
INSERT INTO "Breaking_News" VALUES(4,'Groundbreaking Technology Unveiled','Researchers have unveiled a groundbreaking technology that could revolutionize...',4,5);
INSERT INTO "Breaking_News" VALUES(5,'Health Crisis Worsens Worldwide','The global health crisis continues to worsen, with cases skyrocketing...',5,6);
CREATE TABLE Comments (
    commentId INTEGER PRIMARY KEY,
    text TEXT,
    authorId INTEGER,
    postType TEXT ,
    FOREIGN KEY(authorId) REFERENCES Authors(id));
INSERT INTO "Comments" VALUES(1,'Great job covering the earthquake!',1,'review');
INSERT INTO "Comments" VALUES(2,'Interesting take on the political crisis.',2,'feedback');
INSERT INTO "Comments" VALUES(3,'Important discussion about climate change impacts.',3,'feedback');
INSERT INTO "Comments" VALUES(4,'Well-researched and informative article.',4,'review');
INSERT INTO "Comments" VALUES(5,'Valuable insights into global health issues.',5,'review');
CREATE TABLE Events (
    eventId INTEGER PRIMARY KEY,
    eventName TEXT);
INSERT INTO "Events" VALUES(1,'Turkey–Syria Earthquakes');
INSERT INTO "Events" VALUES(2,'Climate Change Summit');
INSERT INTO "Events" VALUES(3,'Political Crisis in Europe');
INSERT INTO "Events" VALUES(4,'Technology Innovation Conference');
INSERT INTO "Events" VALUES(5,'Global Health Forum');
CREATE TABLE News_Analysis (
    analysisId INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    authorId INTEGER,
    eventId INTEGER,
    FOREIGN KEY(authorId) REFERENCES Authors(id),
    FOREIGN KEY(eventId) REFERENCES Events(eventId));
INSERT INTO "News_Analysis" VALUES(1,'In-Depth Analysis of Turkey–Syria Earthquakes','This article provides an in-depth analysis of the recent earthquakes...',1,1);
INSERT INTO "News_Analysis" VALUES(2,'Examining the Impact of Climate Change on Global Economy','This article examines how climate change is affecting the global economy...',2,3);
INSERT INTO "News_Analysis" VALUES(3,'Analyzing the European Political Crisis','This piece delves into the factors contributing to the European political crisis...',3,4);
INSERT INTO "News_Analysis" VALUES(4,'Assessing Technological Advances in the Past Decade','This article assesses the technological advances made over the past decade...',4,5);
INSERT INTO "News_Analysis" VALUES(5,'Understanding the Global Health Inequalities','This article explores the underlying causes of global health inequalities...',5,6);
CREATE TABLE Sources (
    sourceId INTEGER PRIMARY KEY,
    url TEXT,
    description TEXT,
    eventId INTEGER,
    FOREIGN KEY(eventId) REFERENCES Events(eventId));
INSERT INTO "Sources" VALUES(1,'https://www.example.com/earthquake-news','Latest updates on the Turkey–Syria earthquakes.',1);
INSERT INTO "Sources" VALUES(2,'https://www.example.com/climate-change-summit','Live coverage of the Climate Change Summit.',3);
INSERT INTO "Sources" VALUES(3,'https://www.example.com/political-crisis-in-europe','Updates on the ongoing political crisis in Europe.',4);
INSERT INTO "Sources" VALUES(4,'https://www.example.com/technology-innovation-conference','Highlights from the Technology Innovation Conference.',5);
INSERT INTO "Sources" VALUES(5,'https://www.example.com/global-health-forum','Coverage of the Global Health Forum.',6);
COMMIT;
