package category;

public class MyHeap<T extends Comparable> {

    Comparable[] heap;
    int size;
    int cap;

    public MyHeap() {
        this(0);
    }

    public MyHeap(int cap) {

        heap = new Comparable[cap];
    }

    private void resize(int newCap) {
        if (newCap > size) {
            Comparable[] t = new Comparable[newCap];
            for (int i = 0; i < size; i++) {
                t[i] = heap[i];
            }
            heap = t;
        }
    }

    private boolean less(Comparable[] heap, int a, int b) {
        return heap[a].compareTo(heap[b]) < 0;
    }

    private void swap(Comparable[] heap, int a, int b) {
        Comparable t = heap[a];
        heap[a] = heap[b];
        heap[b] = t;
    }

    private void up(int k) {
        while (k > 0 && less(heap, k / 2, k)) {
            swap(heap, k / 2, k);
            k /= 2;
        }
    }

    private void down(int p) {
        while (p * 2 + 1 < size) {
            int child = p * 2 + 1;
            if (child + 1 < size && less(heap, child, child + 1)) child++;
            if (less(heap, p, child)) {
                swap(heap, p, child);
            }
            p = child;
        }
    }

    public void push(Comparable k) {
        if (size == heap.length) {
            resize(size + 1);
        }
        heap[size++] = k;
        up(size - 1);
    }

    public Comparable poll() {
        Comparable res = heap[0];
        swap(heap, 0, --size);
        heap[size] = null;
        down(0);
        return res;
    }


    //    普通堆排序
    public void heapsort() {
        int[] nums = new int[]{5, 3, 4, 1, 2, 89, -54, 2};
        buildHeap(nums);
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            MyUtile.swap(nums, 0, n - i - 1);
            down(nums, 0, n - i - 1);
        }
        MyUtile.dis(nums);

    }

    public void buildHeap(int[] nums) {
        int n = nums.length;
        for (int i = n / 2 - 1; i >= 0; i--) {
            down(nums, i, n);
        }
    }

    public void down(int[] nums, int p, int end) {
        int child;
        while ((child = p * 2 + 1) < end) {
            if (child + 1 < end && nums[child] < nums[child + 1]) child++;
            if (nums[p] < nums[child]) {
                MyUtile.swap(nums, p, child);
            }
            p = child;
        }
    }


    public static void main(String[] args) {
        MyHeap<Integer> myheap = new MyHeap<>();
        int[] list = new int[]{5, 3, 4, 1, 2, 89, -54, 2};
//        for (int i = 0; i < list.length; i++) {
//            myheap.push(list[i]);
//        }
//        for (int i = 0; i < list.length; i++) {
//            System.out.println(myheap.poll());
//        }
        myheap.heapsort();


    }


}
