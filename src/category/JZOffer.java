package category;

import java.util.Stack;

public class JZOffer {

//    剑指 Offer 09. 用两个栈实现队列
    Stack<Integer> s1=new Stack<Integer>(),s2=new Stack<Integer>();
    public void appendTail(int value) {
        s1.push(value);

    }

    public int deleteHead() {
        if (s1.isEmpty()) {
            return -1;
        }
        while (s1.size() > 1) {
            s2.push( s1.pop());
        }
        int first=s1.pop();
        while (!s2.isEmpty()){
            s1.push(s2.pop());
        }
        return first;
    }




}
