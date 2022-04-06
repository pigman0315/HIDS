#include <iostream>
#include <vector>
#include <stack>
#include <string>

using namespace std;
bool validation(string str){
    stack<char> s;
    bool flg = false;
    int level = 0;
    bool has_n = false;
    for(int i = 0;i < str.length();i++){
        if(str[i] != ']'){
            s.push(str[i]);
            if(str[i] == '[')
                level++;
        }
        else{
            level--;
            while(1){
                char c = s.top();
                if(flg == true && ((c >= 'a' && c <= 'z'))){
                    if(has_n){
                        flg = false;
                        break;
                    }
                    else{
                        return false;
                    }
                }
                else if(flg == false && (c >= '0' && c <= '9')){
                    return false;
                } 
                else if(flg == true && (c >= '0' && c <= '9')){
                    has_n = true;
                }
                else if(c == '['){
                    flg = true;
                }
                s.pop();
                if(s.size() == 0){
                    flg = false;
                    break;
                }
            }
        }
    }
    if(level > 0)
        return false;
    else
        return true;
}

string decode(string input_str){
    if(!validation(input_str)){
        return "ERROR";
    }
    else{
        string output_str;
        stack<char> s;
        string num_str;
        vector<int> num_vec;
        int num = 0;
        int level = 0;
        for(int i = 0;i < input_str.length();i++){
            if(input_str[i] >= '0' && input_str[i] <= '9'){
                num_str.push_back(input_str[i]);
            }
            else if(input_str[i] == '['){
                num = stoi(num_str); // string to integer
                num_str.clear();
                num_vec.push_back(num);
                s.push(input_str[i]);
                level++;
            }
            else if(input_str[i] == ']'){
                string tmp_str;
                while(s.top()!='['){
                    tmp_str.push_back(s.top());
                    s.pop();
                }
                s.pop();
                for(int j = 0;j < num_vec.back();j++){
                    for(int k = tmp_str.length()-1;k >= 0 ;k--){
                        s.push(tmp_str[k]);
                    }
                }
                num_vec.pop_back();
                level--;
            }
            else{
                s.push(input_str[i]);
            }
            if(level == 0){
                int len = s.size();
                string tmp_str2;
                tmp_str2.resize(len);
                for(int i = len-1;i >= 0;i--){
                    tmp_str2[i] = s.top();
                    s.pop();
                }
                output_str += tmp_str2;
            }
        }
        
        return output_str;
    }
    
}

int main(){
    string t1 = "2[b]c";
    string t2 = "1[a]2[b]";
    string t3 = "1[a2[b]";
    string t4 = "3[kk2323]";
    string t5 = "b2[an]a";
    string t6 = "10[bd]2[an]2[sg]";
    string t7 = "go2[d]e3[s]ip";
    string t8 = "2[effp]10[ac2[zn]]";
    string t9 = "1[a[abc]]";
    cout << decode(t1) << endl;
    cout << decode(t2) << endl;
    cout << decode(t3) << endl;
    cout << decode(t4) << endl;
    cout << decode(t5) << endl;
    cout << decode(t6) << endl;
    cout << decode(t7) << endl;
    cout << decode(t8) << endl;
    cout << decode(t9) << endl;
}