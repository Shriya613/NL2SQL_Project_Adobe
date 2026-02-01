BEGIN TRANSACTION;
CREATE TABLE `Address`(
    `address_id` INTEGER PRIMARY KEY,
    `street` TEXT,
    `city` TEXT,
    `state` TEXT,
    `country` TEXT);
INSERT INTO "Address" VALUES(1,'123 Sunset Blvd','Los Angeles','CA','USA');
INSERT INTO "Address" VALUES(2,'456 Central Ave','New York','NY','USA');
INSERT INTO "Address" VALUES(3,'789 Ocean Dr','Miami Beach','FL','USA');
INSERT INTO "Address" VALUES(4,'1314 S Michigan Ave','Chicago','IL','USA');
INSERT INTO "Address" VALUES(5,'901 Main St','Kansas City','MO','USA');
CREATE TABLE `Episode` (
    `episode_id` INTEGER PRIMARY KEY,
    `number` INTEGER,
    `duration` TIME,
    `podcast_id` INTEGER,
    FOREIGN KEY(`podcast_id`) REFERENCES `Podcast`(`podcast_id`));
INSERT INTO "Episode" VALUES(1,1,'01:30:00',1);
INSERT INTO "Episode" VALUES(2,2,'01:00:00',2);
INSERT INTO "Episode" VALUES(3,3,'01:30:00',3);
INSERT INTO "Episode" VALUES(4,4,'01:00:00',4);
INSERT INTO "Episode" VALUES(5,5,'01:30:00',5);
CREATE TABLE `Podcast` (
    `podcast_id` INTEGER PRIMARY KEY,
    `title` TEXT,
    `description` TEXT,
    `release_date` DATE,
    `studio_id` INTEGER,
    `address_id` INTEGER,
    FOREIGN KEY(`studio_id`) REFERENCES `Studio`(`studio_id`),
    FOREIGN KEY(`address_id`) REFERENCES `Address`(`address_id`));
INSERT INTO "Podcast" VALUES(1,'The Morning Show','A daily news and talk show with interviews and reviews of movies, books, music, television shows, etc..','2021-01-01',1,1);
INSERT INTO "Podcast" VALUES(2,'Tech Talk Today','Daily podcast about technology news and events','2021-01-01',2,2);
INSERT INTO "Podcast" VALUES(3,'Breaking Bad','TV series recap and discussion podcast','2021-01-01',3,3);
INSERT INTO "Podcast" VALUES(4,'Late Night With Stephen Colbert','Comedy commentary on current events and pop culture','2021-01-01',4,4);
INSERT INTO "Podcast" VALUES(5,'True Crime Garage','Investigating true crime stories from around the world','2021-01-01',5,5);
CREATE TABLE `Review` (
    `review_id` INTEGER PRIMARY KEY,
    `rating` INTEGER ,
    `comment` TEXT,
    `episode_id` INTEGER,
    `reviewer_id` INTEGER,
    FOREIGN KEY(`episode_id`) REFERENCES `Episode`(`episode_id`),
    FOREIGN KEY(`reviewer_id`) REFERENCES `Reviewer`(`reviewer_id`));
INSERT INTO "Review" VALUES(1,5,'Great episode! Very informative and engaging.',1,1);
INSERT INTO "Review" VALUES(2,4,'Interesting topic but could have been better presented.',2,2);
INSERT INTO "Review" VALUES(3,5,'This is my favorite podcast so far.',3,3);
INSERT INTO "Review" VALUES(4,3,'Could be a bit longer to cover all topics.',4,4);
INSERT INTO "Review" VALUES(5,4,'Very interesting episodes, looking forward to next one.',5,5);
CREATE TABLE `Reviewer` (
    `reviewer_id` INTEGER PRIMARY KEY,
    `username` TEXT,
    `email` TEXT UNIQUE);
INSERT INTO "Reviewer" VALUES(1,'JaneDoe','jane@example.com');
INSERT INTO "Reviewer" VALUES(2,'JohnSmith','john@example.com');
INSERT INTO "Reviewer" VALUES(3,'AliceBrown','alice@example.com');
INSERT INTO "Reviewer" VALUES(4,'RobertWhite','robert@example.com');
INSERT INTO "Reviewer" VALUES(5,'DavidBlack','david@example.com');
CREATE TABLE `Studio` (
    `studio_id` INTEGER PRIMARY KEY,
    `name` TEXT,
    `founded_year` INTEGER);
INSERT INTO "Studio" VALUES(1,'Warner Bros.',2017);
INSERT INTO "Studio" VALUES(2,'Disney',1938);
INSERT INTO "Studio" VALUES(3,'Netflix',1997);
INSERT INTO "Studio" VALUES(4,'Amazon Prime Video',2006);
INSERT INTO "Studio" VALUES(5,'Hulu',2007);
COMMIT;
