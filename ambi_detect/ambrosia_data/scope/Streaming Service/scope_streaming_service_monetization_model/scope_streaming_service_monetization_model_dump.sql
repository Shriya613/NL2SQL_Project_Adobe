BEGIN TRANSACTION;
CREATE TABLE Channel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streaming_service_id INTEGER REFERENCES Streaming_Service(id),
    content_category_id INTEGER REFERENCES Content_Category(id),
    channel_name TEXT,
    channel_description TEXT
);
INSERT INTO "Channel" VALUES(1,1,1,'The Shawshank Redemption','A bank clerk gets sentenced to life in prison.');
INSERT INTO "Channel" VALUES(2,1,2,'Breaking Bad','A high school chemistry teacher turns to a life of crime.');
INSERT INTO "Channel" VALUES(3,2,3,'Blue Planet II','David Attenborough presents a documentary about our world''s oceans.');
INSERT INTO "Channel" VALUES(4,3,4,'Pepper Ann','About a middle school girl who is trying to find her way.');
INSERT INTO "Channel" VALUES(5,4,5,'Dragon Ball Z','A young boy learns martial arts from his mentor after being attacked by evil forces.');
CREATE TABLE Content_Category (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT
);
INSERT INTO "Content_Category" VALUES(1,'Movies');
INSERT INTO "Content_Category" VALUES(2,'TV Shows');
INSERT INTO "Content_Category" VALUES(3,'Documentary');
INSERT INTO "Content_Category" VALUES(4,'Kids');
INSERT INTO "Content_Category" VALUES(5,'Anime');
CREATE TABLE Monetization_Model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT
);
INSERT INTO "Monetization_Model" VALUES(1,'Advertisement');
INSERT INTO "Monetization_Model" VALUES(2,'Pay-per-View');
INSERT INTO "Monetization_Model" VALUES(3,'Free');
INSERT INTO "Monetization_Model" VALUES(4,'Subscription');
INSERT INTO "Monetization_Model" VALUES(5,'Transactional');
CREATE TABLE Streaming_Service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    launch_date DATE
);
INSERT INTO "Streaming_Service" VALUES(1,'Netflix','Leading subscription service that offers a wide variety of award-winning TV shows, movies, anime, documentaries and more.','8/29/2010');
INSERT INTO "Streaming_Service" VALUES(2,'Hulu','Offering hit TV series, hilarious original comedy, can’t-miss events, and films – all in one place.','3/29/2010');
INSERT INTO "Streaming_Service" VALUES(3,'Amazon Prime Video','Unlimited access to thousands of popular movies and TV episodes including Amazon Originals.',2006);
INSERT INTO "Streaming_Service" VALUES(4,'Disney+','Binge on endless stream of Disney, Pixar, Marvel, Star Wars, and National Geographic content.','11/12/2019');
INSERT INTO "Streaming_Service" VALUES(5,'YouTube Premium','Get an ad-free experience of YouTube and YouTube Original Series & Movies.','May 2018');
CREATE TABLE Streaming_Service_Monetization_Model (
    streaming_service_id INTEGER REFERENCES Streaming_Service(id),
    monetization_model_id INTEGER REFERENCES Monetization_Model(id),
    PRIMARY KEY (streaming_service_id, monetization_model_id)
);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(1,1);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(1,2);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(2,3);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(2,4);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(3,5);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(1,4);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(3,4);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(4,4);
INSERT INTO "Streaming_Service_Monetization_Model" VALUES(5,4);
CREATE TABLE Subscription_Plan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streaming_service_id INTEGER REFERENCES Streaming_Service(id),
    price DECIMAL(10, 2) ,
    duration TEXT,
    plan_details TEXT
);
INSERT INTO "Subscription_Plan" VALUES(1,1,7.99,'Monthly','Basic Plan - Unlimited Access to Our Platform');
INSERT INTO "Subscription_Plan" VALUES(2,2,5.99,'Monthly','With Ads Basic Plan - Unlimited Access to Our Platform');
INSERT INTO "Subscription_Plan" VALUES(3,3,6.99,'Yearly','Standard Plan - Unlimited Access to Our Platform');
INSERT INTO "Subscription_Plan" VALUES(4,4,4.99,'Monthly','Mobile Only Plan - Unlimited Access to Our Platform');
INSERT INTO "Subscription_Plan" VALUES(5,5,11.99,'Annual','Premium Plan - No Ads and Download Options Available');
CREATE TABLE Viewer_Stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streaming_service_id INTEGER REFERENCES Streaming_Service(id),
    channel_id INTEGER REFERENCES Channel(id),
    views INTEGER DEFAULT 0,
    unique_viewers INTEGER DEFAULT 0
);
INSERT INTO "Viewer_Stats" VALUES(1,1,1,1000,300);
INSERT INTO "Viewer_Stats" VALUES(2,1,2,2000,500);
INSERT INTO "Viewer_Stats" VALUES(3,2,3,1500,450);
INSERT INTO "Viewer_Stats" VALUES(4,2,4,1200,400);
INSERT INTO "Viewer_Stats" VALUES(5,3,5,800,250);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Monetization_Model',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Service',5);
INSERT INTO "sqlite_sequence" VALUES('Content_Category',5);
INSERT INTO "sqlite_sequence" VALUES('Channel',5);
INSERT INTO "sqlite_sequence" VALUES('Subscription_Plan',5);
INSERT INTO "sqlite_sequence" VALUES('Viewer_Stats',5);
COMMIT;
