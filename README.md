# Resolution-Free Human Image Generation and Control
We aim to build an image generation model that builds upon DiT and Patch n' Pack, which can generate human images with high resolution and control over the generated images. 


## Team Members
| Github ID | Name | email  | Phone num.  | School. | Role | 
|:--------: |:----:|:--------:|:---------:|:----:| :----:|
| three0-s  | Yewon Lim | lim473@purdue.edu | +1 (765) 412-6118 | Yonsei Univ. C.S.| Director |
| Haileyyyyyyyy | Dayeon Kim  | kim4732@purdue.edu | +1 (765) 694-5879 | Sookmyung W. Univ. S.C. | Planning Leader |
| jeahoyang | Jaeho Yang | yang2731@purdue.edu | +1 (765) 767-3162 |  Yonsei Univ. C.E. | Testing Leader |
| sunkite3-3 | Haeyeon Kim | kim4733@purdue.edu | +1 (765) 479-2799 |  Kyonggi Univ. AI.C.E.  | Documentation Leader |
| uujeonglee | Yoojeong lee | lee5202@purdue.edu | +1 (765) 767-2172 | Sookmyung W. Univ. IT.E. | QA Leader |


## Study Plans
| #   | Date    | Time | Topic            | Reviewer     | Note                                    |
|-----|---------|------|------------------|--------------|-----------------------------------------|
| 1   | 3/20 Wed| 4:00 | VGG          | Yewon Lim    | [VGG](#note1)               |
| 2   | 3/21 Thu| 14:00 | ResNet           | Haeyeon Kim   | [ResNet](https://github.com/three0-s/huchudle/blob/develop/logs/notes/ResNet.pdf)               |
| 3   | 3/22 Fri| 15:30 | seq2seq          | Yoojeong Lee   | [Link to Note 3](https://github.com/three0-s/huchudle/blob/develop/logs/notes/seq2seq.pdf)               |
| 4   | 3/25 Mon| 15:30 | GRU | Dayeon Kim  | [Link to Note 4](#note4)               |
| 5   | 3/26 Tue| 13:30 | Transformer        | Jaeho Yang | [Link to Note 5](#note5)               |
| 6   | 3/27 Wed| 15:30 | ViT       | Haeyeon Kim | [Link to Note 5](#note5)        
| 7   | 3/28 Thu| 15:30 |  MAE         | Yoojeong Lee | [Link to Note 5](#note5)        |
| 8   | 3/29 Fri| 15:30 | VAE         | Yewon Lim  | [Link to Note 5](#note5)               |
| 9   | 4/1 Mon| 15:30 | GAN         | Dayeon Kim  | [Link to Note 6](#note6)               |
| 10   | 4/2 Tue| 15:30 | DDPM       | Jaeho Yang   | [Link to Note 7](#note7)               |
| 11   | 4/3 Wed| 15:30 | DDIM      | Haeyeon Kim  | [Link to Note 8](#note8)               | 
| 12   | 4/4 Thu| 15:30 | DiT      | Yoojeong Lee  | [Link to Note 8](#note8)               | 
| 13 | 4/5 Fri| 15:30 | NaViT      | Dayeon Kim  | [Link to Note 8](#note8)               | 


**Commit Convention**

```
# Write the title below up to 50 characters: ex) Feat: Add Key mapping  
# Write the body below  
# Write the footer below: ex) Github issue #23  

# --- commit end ---  
# <type> list
# feat : function (new feature)  
# fix: bug (bug fix)  
# refactor : refactoring  
# design: change user UI design, such as CSS, etc.  
# comment: add and change necessary comments  
# style : style (code format, add semicolon: no change in business logic)  
# docs: Modify documentation (add, modify, delete documentation, README)  
# test : tests (add, modify, delete test code: no changes to business logic)  
# chore: miscellaneous changes (build script modifications, assets, package manager, etc.)  
# init : Initial creation  
# rename : file or folder renaming or moving only  
# remove : if the only action performed is to delete a file  
# ------------------  
# capitalize the first letter of the title  
# Title is a command statement  
# No periods (.) at the end of the title  
# Separate title and body by a single line  
# The body explains the "what" and "why" rather than the "how".  
# Separate multi-line messages in the body with a "-"  
# ------------------  
# <Footer>  
# optioanl not required  
# Fixes: issue is being fixed (if not already resolved)  
# Resolves: used when the issue has been resolved  
# Ref: used when there is an issue to refer to  
# Related to : Issue number related to this commit (if not already resolved)  
# ex) Fixes: #47 Related to: #32, #21 

```